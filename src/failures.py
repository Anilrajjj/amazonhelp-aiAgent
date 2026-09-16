"""Failure analysis: pull the real errors out of a system's predictions.

Groups mistakes into the categories that matter operationally rather than
listing them one by one, and prints real messages for each so the report can
quote them. Nothing here is hand-picked: the groups are computed and the
examples are the first N in each group.

Usage:  python src/failures.py --system agent [--top 5]
"""
import argparse, collections, json, os

import console
from dataset import ROOT, is_english
from intents import ALWAYS_ESCALATE_INTENTS

RESULTS = os.path.join(ROOT, "results")


def load(system):
    path = os.path.join(RESULTS, f"preds_{system}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found -- run evaluate.py --system {system}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def categorise(r):
    """Return the list of failure tags for one prediction."""
    tags = []
    gi, pi = r["gold_intent"], r["pred"].get("intent")
    gr, pr = r["gold_route"], r["pred"].get("route")

    if gi != pi:
        tags.append(f"intent:{gi}->{pi}")
    if gr == "escalate" and pr == "auto":
        tags.append(f"MISSED_ESCALATION:{r['gold_escalation_reason']}")
    if gr == "auto" and pr == "escalate":
        tags.append(f"over_escalation:{r['pred'].get('escalation_reason')}")
    if not is_english(r["customer_message"]):
        tags.append("non_english_leaked_into_scope")
    return tags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", default="agent")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()
    console.init()

    rows = load(a.system)
    buckets = collections.defaultdict(list)
    for r in rows:
        for t in categorise(r):
            buckets[t].append(r)

    # Roll the many intent:X->Y tags up into one family so the ranking is not
    # dominated by a long tail of one-off confusions.
    families = collections.defaultdict(list)
    for tag, rs in buckets.items():
        fam = ("missed escalation" if tag.startswith("MISSED_ESCALATION") else
               "over-escalation" if tag.startswith("over_escalation") else
               "non-English leaked into scope" if tag.startswith("non_english") else
               "intent confusion")
        families[fam].extend(rs)

    print(f"\n{'=' * 74}\nFAILURE ANALYSIS -- {a.system}   (n={len(rows)})\n{'=' * 74}")
    print("\nby family:")
    for fam, rs in sorted(families.items(), key=lambda kv: -len(kv[1])):
        uniq = {r["conversation_id"] for r in rs}
        print(f"  {fam:<34}{len(uniq):>4}")

    print(f"\n{'-' * 74}\nTOP {a.top} SPECIFIC FAILURE MODES\n{'-' * 74}")
    for tag, rs in sorted(buckets.items(), key=lambda kv: -len(kv[1]))[:a.top]:
        print(f"\n### {tag}   ({len(rs)} cases)")
        for r in rs[:a.examples]:
            p = r["pred"]
            print(f"  C: {r['customer_message'][:190]}")
            print(f"     gold  intent={r['gold_intent']} route={r['gold_route']}"
                  f" reason={r['gold_escalation_reason']}")
            print(f"     pred  intent={p.get('intent')} route={p.get('route')}"
                  f" reason={p.get('escalation_reason')} conf={p.get('confidence')}")
            if p.get("reply"):
                print(f"     reply: {p['reply'][:160]}")
            print()

    # Confusion pairs, most frequent first -- what the taxonomy is failing to
    # separate, as opposed to what the model is failing to learn.
    print(f"{'-' * 74}\nINTENT CONFUSION PAIRS\n{'-' * 74}")
    pairs = collections.Counter(
        (r["gold_intent"], r["pred"].get("intent")) for r in rows
        if r["gold_intent"] != r["pred"].get("intent"))
    for (g, p), n in pairs.most_common(10):
        print(f"  {n:>3}  {g}  ->  {p}")

    # Where the policy itself, not the model, caused the escalation.
    policy = [r for r in rows
              if r["pred"].get("escalation_reason") == "account_security_policy"]
    print(f"\n{len(policy)} escalations came from the always-escalate intent policy "
          f"({', '.join(ALWAYS_ESCALATE_INTENTS)}), not from the model.")


if __name__ == "__main__":
    main()
