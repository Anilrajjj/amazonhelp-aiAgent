"""Automated evaluation harness.

Runs a system over data/golden.jsonl and reports three families of metrics:

INTENT   accuracy and macro-F1 over the eight labels. Macro-F1 is the headline,
         because the golden set is stratified and accuracy on a stratified
         sample says almost nothing.

ROUTING  accuracy, plus the two error types kept separate on purpose. A missed
         escalation (gold=escalate, predicted=auto) sends an unreviewed reply to
         someone reporting fraud or threatening legal action. An over-escalation
         (gold=auto, predicted=escalate) costs an agent thirty seconds. Reporting
         one routing accuracy number hides that asymmetry, so this prints both
         and treats missed escalations as the number that matters.

REPLY    deterministic safety checks only. Whether a reply is *good* is the
         judge's job (judge.py); what is checked here is whether it is *unsafe*,
         which is decidable with regexes: invented URLs, invented facts the brand
         cannot know from Twitter, and promised timeframes.

Usage:
    python src/evaluate.py --system trivial
    python src/evaluate.py --system simple
    python src/evaluate.py --system agent        (needs an LLM key)
    python src/evaluate.py --system all
"""
import argparse, collections, json, os, re, sys

from sklearn.metrics import classification_report, f1_score

import console
from dataset import ROOT, load_split
from intents import LABELS
from retrieval import PrecedentIndex

GOLDEN = os.path.join(ROOT, "data", "golden.jsonl")
RESULTS = os.path.join(ROOT, "results")

# --- reply safety checks -----------------------------------------------------

URL_RE = re.compile(r"https?://|www\.|\b[a-z0-9-]+\.(com|co\.uk|in|net|org)\b", re.I)
# A concrete promise the brand cannot keep from a public Twitter reply.
PROMISE_RE = re.compile(
    r"\b(within \d+|in \d+ (hours?|days?|minutes?)|by (tomorrow|today|monday|"
    r"tuesday|wednesday|thursday|friday|saturday|sunday)|next \d+ (hours?|days?)|"
    r"will (be )?(deliver|arrive|refund|credit)\w*\s+(it|your|the)?\s*(by|on|within)|"
    r"we have (refunded|issued|shipped|cancelled)|has been (refunded|shipped|"
    r"dispatched|delivered)|i have (refunded|issued|cancelled))\b", re.I)
# Order/tracking numbers or money amounts stated by the agent.
INVENTED_FACT_RE = re.compile(r"\b\d{3}-\d{7}-\d{7}\b|[£$€]\s?\d+|\b\d+\.\d{2}\b")


def check_reply(reply):
    """Return the set of safety violations in a drafted reply."""
    if not reply:
        return {"empty"}
    bad = set()
    # <link> is the sanctioned placeholder; anything else URL-shaped is invented
    stripped = reply.replace("<link>", " ")
    if URL_RE.search(stripped):
        bad.add("invented_url")
    if PROMISE_RE.search(reply):
        bad.add("promised_outcome")
    if INVENTED_FACT_RE.search(reply):
        bad.add("invented_fact")
    if len(reply) > 400:
        bad.add("too_long")
    return bad


# --- metrics -----------------------------------------------------------------

def intent_metrics(gold, pred):
    present = [l for l in LABELS if l in set(gold) | set(pred)]
    return {
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold),
        "macro_f1": f1_score(gold, pred, labels=present, average="macro", zero_division=0),
        "report": classification_report(gold, pred, labels=present,
                                        zero_division=0, digits=3),
    }


def routing_metrics(gold, pred):
    n = len(gold)
    missed = sum(g == "escalate" and p == "auto" for g, p in zip(gold, pred))
    over = sum(g == "auto" and p == "escalate" for g, p in zip(gold, pred))
    gold_esc = sum(g == "escalate" for g in gold)
    gold_auto = n - gold_esc
    pred_esc = sum(p == "escalate" for p in pred)
    caught = gold_esc - missed
    return {
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / n,
        "escalation_recall": caught / gold_esc if gold_esc else 0.0,
        "escalation_precision": caught / pred_esc if pred_esc else 0.0,
        "missed_escalations": missed,
        "missed_escalation_rate": missed / gold_esc if gold_esc else 0.0,
        "over_escalations": over,
        "over_escalation_rate": over / gold_auto if gold_auto else 0.0,
        "predicted_escalation_share": pred_esc / n,
    }


def reply_metrics(rows):
    viol = collections.Counter()
    checked = 0
    handoff = 0
    lengths = []
    for r in rows:
        reply = r["pred"].get("reply")
        if r["pred"].get("route") == "escalate" and not reply:
            continue   # escalated with no draft is legitimate
        checked += 1
        for v in check_reply(reply):
            viol[v] += 1
        if reply:
            lengths.append(len(reply))
            if "<link>" in reply:
                handoff += 1
    clean = checked - sum(1 for r in rows
                          if r["pred"].get("reply") is not None
                          and check_reply(r["pred"].get("reply")))
    return {
        "replies_checked": checked,
        "clean_rate": (checked - sum(v for k, v in viol.items()
                                     if k in ("invented_url", "promised_outcome",
                                              "invented_fact", "empty"))) / checked
        if checked else 0.0,
        "violations": dict(viol),
        "handoff_rate": handoff / checked if checked else 0.0,
        "mean_length": sum(lengths) / len(lengths) if lengths else 0,
    }


# --- runners -----------------------------------------------------------------

def load_golden():
    with open(GOLDEN, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def run_system(name, rows, corpus, index):
    """Return a list of {gold..., pred...} records."""
    if name == "trivial":
        from baselines import TrivialBaseline
        sysobj = TrivialBaseline()
        handle = lambda m: sysobj.handle(m)
    elif name == "simple":
        from baselines import SimpleBaseline
        print("  training simple baseline ...", flush=True)
        sysobj = SimpleBaseline(corpus, index)
        handle = lambda m: sysobj.handle(m)
    elif name == "agent":
        import agent, llm
        if not llm.is_available():
            raise SystemExit(
                "system 'agent' needs an LLM. Set OPENROUTER_API_KEY / "
                "GROQ_API_KEY / NVIDIA_API_KEY.")
        handle = lambda m: agent.handle(m, index)
    else:
        raise SystemExit(f"unknown system {name!r}")

    out = []
    for i, r in enumerate(rows, 1):
        if name == "agent" and i % 25 == 0:
            print(f"    {i}/{len(rows)}", flush=True)
        out.append({
            "conversation_id": r["conversation_id"],
            "customer_message": r["customer_message"],
            "gold_intent": r["gold_intent"],
            "gold_route": r["gold_route"],
            "gold_escalation_reason": r["gold_escalation_reason"],
            "brand_reply": r.get("brand_reply"),
            "pred": handle(r["customer_message"]),
        })
    return out


def score(name, records):
    gi = [r["gold_intent"] for r in records]
    pi = [r["pred"].get("intent") or "feedback_or_other" for r in records]
    gr = [r["gold_route"] for r in records]
    pr = [r["pred"].get("route") or "escalate" for r in records]
    return {
        "system": name,
        "n": len(records),
        "intent": intent_metrics(gi, pi),
        "routing": routing_metrics(gr, pr),
        "reply": reply_metrics(records),
    }


def print_scores(s):
    print(f"\n{'=' * 72}\n{s['system'].upper()}  (n={s['n']})\n{'=' * 72}")
    i, r, p = s["intent"], s["routing"], s["reply"]
    print(f"INTENT   accuracy {i['accuracy']:.3f}   macro-F1 {i['macro_f1']:.3f}")
    print(i["report"])
    print(f"ROUTING  accuracy {r['accuracy']:.3f}")
    print(f"         escalation recall    {r['escalation_recall']:.3f}  "
          f"precision {r['escalation_precision']:.3f}")
    print(f"         MISSED escalations   {r['missed_escalations']:>3}  "
          f"({r['missed_escalation_rate']:.1%} of cases that needed a human)")
    print(f"         over-escalations     {r['over_escalations']:>3}  "
          f"({r['over_escalation_rate']:.1%} of cases that did not)")
    print(f"         predicted escalation share {r['predicted_escalation_share']:.1%}")
    print(f"REPLY    checked {p['replies_checked']}   "
          f"safety-clean {p['clean_rate']:.3f}   "
          f"handoff <link> {p['handoff_rate']:.1%}   "
          f"mean len {p['mean_length']:.0f}")
    print(f"         violations: {p['violations'] or 'none'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="all",
                    choices=["trivial", "simple", "agent", "all"])
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate on the first N golden examples only")
    a = ap.parse_args()
    console.init()

    rows = load_golden()
    if a.limit:
        rows = rows[:a.limit]
    corpus, _ = load_split()
    print(f"golden set {len(rows)}   retrieval corpus {len(corpus):,}")
    print("building precedent index ...", flush=True)
    index = PrecedentIndex(corpus)

    systems = ["trivial", "simple", "agent"] if a.system == "all" else [a.system]
    os.makedirs(RESULTS, exist_ok=True)
    summary = []
    for name in systems:
        print(f"\nrunning {name} ...", flush=True)
        try:
            records = run_system(name, rows, corpus, index)
        except SystemExit as e:
            print(f"  skipped: {e}")
            continue
        with open(os.path.join(RESULTS, f"preds_{name}.jsonl"), "w",
                  encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        s = score(name, records)
        print_scores(s)
        summary.append({k: v for k, v in s.items() if k != "intent"} |
                       {"intent": {k: v for k, v in s["intent"].items()
                                   if k != "report"}})

    with open(os.path.join(RESULTS, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\npredictions + metrics written to {RESULTS}/")


if __name__ == "__main__":
    main()
