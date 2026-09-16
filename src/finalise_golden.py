"""Join the hand-assigned labels onto the frozen sample -> data/golden.jsonl.

Joins on conversation_id and fails loudly if anything does not line up. An
earlier index-based version silently mislabelled 88 of 220 rows when the sample
changed underneath it, so every mismatch here is an assertion, not a warning.
"""
import collections, json, os

import console
from dataset import ROOT
from _golden_labels import LABELS
from intents import LABELS as INTENT_LABELS

SAMPLE = os.path.join(ROOT, "data", "golden_unlabelled.jsonl")
OUT = os.path.join(ROOT, "data", "golden.jsonl")


def main():
    console.init()
    with open(SAMPLE, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]

    sampled = {r["conversation_id"] for r in rows}
    labelled = set(LABELS)
    missing = sampled - labelled
    extra = labelled - sampled
    assert not missing, f"{len(missing)} sampled messages have no label: {sorted(missing)[:5]}"
    assert not extra, f"{len(extra)} labels have no sampled message: {sorted(extra)[:5]}"

    out = []
    for r in rows:
        intent, route, reason = LABELS[r["conversation_id"]]
        assert intent in INTENT_LABELS, f"{r['conversation_id']}: unknown intent {intent}"
        assert route in ("auto", "escalate"), f"{r['conversation_id']}: bad route {route}"
        assert (reason is None) == (route == "auto"), \
            f"{r['conversation_id']}: reason/route mismatch"
        out.append(dict(r, gold_intent=intent, gold_route=route,
                        gold_escalation_reason=reason))

    with open(OUT, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    ic = collections.Counter(r["gold_intent"] for r in out)
    rc = collections.Counter(r["gold_route"] for r in out)
    xc = collections.Counter(r["gold_escalation_reason"] for r in out
                             if r["gold_escalation_reason"])
    print(f"wrote {len(out)} labelled examples -> data/golden.jsonl\n")
    print("intent distribution:")
    for k, v in ic.most_common():
        print(f"  {k:<24}{v:>4}  {100 * v / len(out):5.1f}%")
    print("\nroute distribution:")
    for k, v in rc.most_common():
        print(f"  {k:<24}{v:>4}  {100 * v / len(out):5.1f}%")
    print("\nescalation reasons:")
    for k, v in xc.most_common():
        print(f"  {k:<28}{v:>4}")


if __name__ == "__main__":
    main()
