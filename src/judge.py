"""LLM-as-judge for reply quality, and the evidence that the judge can be trusted.

The rubric scores four things a support reply either does or does not do. Each
is scored 1-5 with the anchors written into the prompt, plus one binary
"send_without_review" verdict, which is the only number used for
judge-vs-human agreement -- a binary decision is what the system actually acts
on, and agreement on a 5-point scale would flatter the judge.

An LLM judge is not evidence on its own. `--agreement` compares the judge
against the human ratings in _judge_human.py on the same replies and reports
raw agreement and Cohen's kappa. Kappa matters more than raw agreement here
because the classes are unbalanced: a judge that says "send" every time would
score ~0.75 raw and ~0.0 kappa.

Usage:
    python src/judge.py --sample          # emit replies for a human to rate
    python src/judge.py --run   simple    # score a system's replies
    python src/judge.py --agreement       # judge vs human: raw + Cohen's kappa
"""
import argparse, json, os, random

import console
import llm
from dataset import ROOT

RESULTS = os.path.join(ROOT, "results")
SAMPLE_PATH = os.path.join(RESULTS, "judge_sample.jsonl")

SYSTEM = ("You are a strict quality reviewer for customer-support replies. "
          "You judge only what is in front of you and you never reward a reply "
          "for sounding pleasant if it fails the customer.")

RUBRIC = """Score a drafted public reply from Amazon's Twitter support account.

Context you must apply:
- AmazonHelp cannot see orders or accounts from Twitter. A reply that states a delivery date, refund amount, order status or any account fact is fabricating.
- The house style is one or two sentences: acknowledge the specific problem, then move the customer to a private channel. "<link>" is a sanctioned placeholder for a real URL.

CUSTOMER MESSAGE:
\"\"\"{message}\"\"\"

DRAFTED REPLY:
\"\"\"{reply}\"\"\"

Score each 1-5:
- relevance: does it address the specific problem this customer described? (1 = generic or wrong problem, 3 = right topic but generic, 5 = names their actual problem)
- grounded: is everything in it true-by-construction, i.e. does it avoid asserting facts the brand cannot know from Twitter? (1 = invents order/refund/delivery facts, 3 = vague but harmless, 5 = asserts nothing it cannot know)
- tone: does it match a calm, apologetic, professional support voice? (1 = cold, robotic or flippant, 5 = natural and appropriate to how upset they are)
- actionable: does the customer know what happens next? (1 = no next step, 3 = vague, 5 = a clear next step or a clear handoff)

Then decide:
- send_without_review: true only if you would let this go out publicly with no human reading it first. Any fabricated fact, any promise the brand cannot keep, or any reply that would anger an already-angry customer is false.

Return ONLY JSON:
{{"relevance": <1-5>, "grounded": <1-5>, "tone": <1-5>, "actionable": <1-5>,
  "send_without_review": <true|false>, "why": "<one short sentence>"}}"""


def judge_one(message, reply):
    return llm.chat_json(RUBRIC.format(message=message.replace('"""', "'''"),
                                       reply=(reply or "").replace('"""', "'''")),
                         system=SYSTEM, max_tokens=900)


# --- sampling for human rating ----------------------------------------------

def build_sample(systems, n_per_system=25, seed=5):
    """Draw replies for a human to rate. Drawn from every system that has run,
    so the judge is validated on the full quality range it will see, not only on
    the good replies."""
    rng = random.Random(seed)
    out = []
    for name in systems:
        path = os.path.join(RESULTS, f"preds_{name}.jsonl")
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        rows = [r for r in rows if r["pred"].get("reply")]
        for r in rng.sample(rows, min(n_per_system, len(rows))):
            out.append({
                "system": name,
                "conversation_id": r["conversation_id"],
                "customer_message": r["customer_message"],
                "reply": r["pred"]["reply"],
            })
    rng.shuffle(out)   # so the rater cannot tell which system produced what
    os.makedirs(RESULTS, exist_ok=True)
    with open(SAMPLE_PATH, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


# --- agreement ---------------------------------------------------------------

def cohen_kappa(a, b):
    """Cohen's kappa for two binary raters, computed directly so the metric is
    auditable rather than a library call."""
    n = len(a)
    assert n == len(b) and n > 0
    agree = sum(x == y for x, y in zip(a, b)) / n
    pa_true, pb_true = sum(a) / n, sum(b) / n
    chance = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if chance == 1.0:
        return agree, chance, 0.0
    return agree, chance, (agree - chance) / (1 - chance)


def agreement():
    from _judge_human import HUMAN_RATINGS
    rows = [json.loads(l) for l in open(SAMPLE_PATH, encoding="utf-8") if l.strip()]
    assert len(HUMAN_RATINGS) == len(rows), \
        f"{len(HUMAN_RATINGS)} human ratings for {len(rows)} sampled replies"

    human, judge, disagreements = [], [], []
    for i, (row, hv) in enumerate(zip(rows, HUMAN_RATINGS)):
        v = judge_one(row["customer_message"], row["reply"])
        jv = bool(v.get("send_without_review"))
        human.append(bool(hv))
        judge.append(jv)
        if bool(hv) != jv:
            disagreements.append({"i": i, "system": row["system"],
                                  "human": bool(hv), "judge": jv,
                                  "why": v.get("why"),
                                  "message": row["customer_message"][:150],
                                  "reply": row["reply"][:150]})

    raw, chance, k = cohen_kappa(human, judge)
    n = len(human)
    print(f"\njudge vs human on 'send without review'  (n={n})")
    print(f"  human says send : {sum(human):>3} / {n}")
    print(f"  judge says send : {sum(judge):>3} / {n}")
    print(f"  raw agreement   : {raw:.3f}")
    print(f"  chance agreement: {chance:.3f}")
    print(f"  Cohen's kappa   : {k:.3f}")
    print(f"  judge too lenient (judge send, human not): "
          f"{sum(1 for d in disagreements if d['judge'] and not d['human'])}")
    print(f"  judge too strict  (human send, judge not): "
          f"{sum(1 for d in disagreements if d['human'] and not d['judge'])}")

    with open(os.path.join(RESULTS, "judge_agreement.json"), "w", encoding="utf-8") as f:
        json.dump({"n": n, "raw_agreement": raw, "chance_agreement": chance,
                   "cohens_kappa": k, "disagreements": disagreements}, f, indent=2)
    print(f"\n{len(disagreements)} disagreements written to results/judge_agreement.json")
    for d in disagreements[:6]:
        print(f"\n  [{d['system']}] human={d['human']} judge={d['judge']}  {d['why']}")
        print(f"    C: {d['message']}")
        print(f"    A: {d['reply']}")


def run(system):
    path = os.path.join(RESULTS, f"preds_{system}.jsonl")
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    scored, sends = [], 0
    for i, r in enumerate(rows, 1):
        if not r["pred"].get("reply"):
            continue
        if i % 25 == 0:
            print(f"  {i}/{len(rows)}", flush=True)
        v = judge_one(r["customer_message"], r["pred"]["reply"])
        scored.append(v)
        sends += bool(v.get("send_without_review"))
    if not scored:
        print("no replies to judge")
        return
    keys = ["relevance", "grounded", "tone", "actionable"]
    means = {k: sum(float(s.get(k) or 0) for s in scored) / len(scored) for k in keys}
    print(f"\nJUDGE  {system}  (n={len(scored)})")
    for k in keys:
        print(f"  {k:<12}{means[k]:.2f} / 5")
    print(f"  send_without_review  {sends}/{len(scored)} = {sends/len(scored):.1%}")
    with open(os.path.join(RESULTS, f"judge_{system}.json"), "w", encoding="utf-8") as f:
        json.dump({"system": system, "n": len(scored), "means": means,
                   "send_rate": sends / len(scored), "scores": scored}, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", action="store_true",
                    help="emit results/judge_sample.jsonl for a human to rate")
    ap.add_argument("--run", help="judge a system's replies (trivial|simple|agent)")
    ap.add_argument("--agreement", action="store_true",
                    help="compare judge to the human ratings in _judge_human.py")
    a = ap.parse_args()
    console.init()

    if a.sample:
        rows = build_sample(["trivial", "simple", "agent"])
        print(f"wrote {len(rows)} replies to {SAMPLE_PATH} for human rating")
        return
    if not llm.is_available():
        raise SystemExit("judge needs an LLM. Set OPENROUTER_API_KEY / "
                         "GROQ_API_KEY / NVIDIA_API_KEY.")
    if a.run:
        run(a.run)
    if a.agreement:
        agreement()


if __name__ == "__main__":
    main()
