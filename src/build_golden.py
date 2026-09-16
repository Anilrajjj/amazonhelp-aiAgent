"""Sample the messages that go into the hand-labelled golden set.

Sampling note (also reproduced in the README):

* Source is the held-out 20% eval pool only, so nothing in the golden set can be
  retrieved as its own precedent.
* A pure random sample of 200 would contain ~12 account_access messages and ~1
  or 2 of some escalation triggers -- too few to measure anything. So the sample
  is stratified: a fixed quota per intent drawn from a cheap keyword prior,
  plus a quota of messages the prior cannot place at all, plus a quota of
  messages that hit an escalation trigger.
* The keyword prior only chooses *what to read*. Every label is then assigned by
  hand against the written definitions in src/intents.py, and the prior's guess
  is deliberately not shown during labelling.
* Stratifying makes per-class metrics meaningful but makes the sample's class
  mix unrepresentative of live traffic. Macro-F1 is therefore the headline, and
  the README's "what is misleading" section reweights to the natural prior.
"""
import argparse, json, os, random, re
import console
from dataset import load_split, ROOT
from intents import LABELS, resolve_conflict, ESCALATE_RULES

# Cheap prior used ONLY to stratify the reading list, never as a label.
PRIOR_PATTERNS = {
    "delivery_issue": r"deliver|package|parcel|courier|carrier|shipped|shipping|arriv|tracking|lost|late|missing",
    "order_status": r"\border\b|dispatch|when will|status|cancel my order|change my order|preorder|pre-order",
    "refund_return": r"refund|return|replacement|exchange|money back|reimburs",
    "payment_billing": r"charg|payment|card|billed|billing|invoice|price|paid|\bpay\b|gift card|discount|coupon",
    "prime_membership": r"prime|membership|subscription|subscribe|renew",
    "account_access": r"account|login|log in|sign in|password|hacked|locked|two-step|verification|email address",
    "product_or_app_issue": r"\bapp\b|website|site|error|crash|kindle|echo|alexa|fire ?stick|prime video|episode|broken|defective|damaged|fake|counterfeit|wrong item|not work",
}

QUOTAS = {
    "delivery_issue": 34,
    "order_status": 26,
    "refund_return": 24,
    "payment_billing": 24,
    "prime_membership": 22,
    "account_access": 22,
    "product_or_app_issue": 24,
    "_no_prior": 24,       # messages the prior cannot place -- praise, rants, vague
    "_escalation": 20,     # messages that trip an escalation trigger
}


def prior_label(text):
    t = text.lower()
    hits = [k for k, p in PRIOR_PATTERNS.items() if re.search(p, t)]
    return resolve_conflict(hits) if hits else None


def trips_escalation(text):
    t = text.lower()
    return [name for name, pat in ESCALATE_RULES if re.search(pat, t)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "golden_unlabelled.jsonl"))
    ap.add_argument("--seed", type=int, default=20240917)
    a = ap.parse_args()
    console.init()

    _, pool = load_split()
    rng = random.Random(a.seed)
    rng.shuffle(pool)

    buckets = {k: [] for k in QUOTAS}
    for r in pool:
        msg = r["customer_message"]
        # keep messages long enough to actually carry an intent
        if not (25 <= len(msg) <= 400):
            continue
        if trips_escalation(msg) and len(buckets["_escalation"]) < QUOTAS["_escalation"]:
            buckets["_escalation"].append(r)
            continue
        p = prior_label(msg)
        key = p if p else "_no_prior"
        if len(buckets[key]) < QUOTAS[key]:
            buckets[key].append(r)
        if all(len(buckets[k]) >= QUOTAS[k] for k in QUOTAS):
            break

    picked, seen = [], set()
    for key, rows in buckets.items():
        for r in rows:
            if r["conversation_id"] in seen:
                continue
            seen.add(r["conversation_id"])
            picked.append({
                "conversation_id": r["conversation_id"],
                "customer_message": r["customer_message"],
                "brand_reply": r["brand_reply"],
                "sampled_from": key,
            })
    rng.shuffle(picked)   # so labelling order does not follow the strata

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in picked:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"eval pool {len(pool):,} -> sampled {len(picked)} for labelling -> {a.out}")
    for k in QUOTAS:
        print(f"  {k:<24}{len(buckets[k]):>4} / {QUOTAS[k]}")


if __name__ == "__main__":
    main()
