"""Extract AmazonHelp support conversations from the raw twcs.csv dump.

Two passes over the CSV:
  1. build the reply graph (tweet -> parent) for every tweet, and note which
     tweets AmazonHelp wrote;
  2. re-read and keep the full text only for tweets in a connected component
     that AmazonHelp took part in.

The two-pass shape matters: a customer's opening tweet often does not contain
the "@AmazonHelp" handle (the brand picks the complaint up itself), so filtering
on the handle in one pass silently truncates the front of the conversation.

Writes data/amazonhelp_threads.jsonl -- one conversation per line.
Usage: python src/prepare_data.py [--csv twcs.csv] [--out ...]
"""
import argparse, csv, json, os, re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

csv.field_size_limit(10 ** 7)

BRAND = "AmazonHelp"
MENTION_RE = re.compile(r"@\w+")
URL_RE = re.compile(r"https?://\S+")
SIG_RE = re.compile(r"\^\s*\w{1,4}\s*$")  # agent initials sign-off, e.g. "^HS"


def clean(text, strip_signature=False):
    """Strip @mentions and URLs, collapse whitespace.

    Keeps the message readable without leaking the brand handle or the
    per-customer numeric ids (which are pseudonyms, not content)."""
    t = URL_RE.sub(" <link> ", text)
    t = MENTION_RE.sub(" ", t)
    t = t.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    t = re.sub(r"\s+", " ", t).strip()
    if strip_signature:
        t = SIG_RE.sub("", t).strip()
    return t


class Union:
    """Union-find over tweet ids, used to group a thread into one component."""

    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        root = x
        while self.p[root] != root:
            root = self.p[root]
        while self.p[x] != root:
            self.p[x], x = root, self.p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def pass1(csv_path):
    """Return (component_root -> True for brand components, union structure)."""
    uf = Union()
    brand_tweets = []
    n = 0
    with open(csv_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            n += 1
            tid = r["tweet_id"]
            uf.find(tid)
            p = r["in_response_to_tweet_id"]
            if p:
                uf.union(tid, p)
            if r["author_id"] == BRAND:
                brand_tweets.append(tid)
    brand_roots = {uf.find(t) for t in brand_tweets}
    return uf, brand_roots, n


def pass2(csv_path, uf, brand_roots):
    """Collect full rows for tweets in AmazonHelp components, grouped by thread."""
    threads = defaultdict(list)
    with open(csv_path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            root = uf.find(r["tweet_id"])
            if root not in brand_roots:
                continue
            threads[root].append({
                "tweet_id": r["tweet_id"],
                "author_id": r["author_id"],
                "inbound": r["inbound"] == "True",
                "created_at": r["created_at"],
                "text": r["text"],
                "parent": r["in_response_to_tweet_id"] or None,
            })
    return threads


def order_thread(msgs):
    """Order a thread by hops back to its root, then by tweet id.

    in_response_to_tweet_id points at the tweet being replied to, so following
    parents walks backwards to the conversation opener; hop count is a stable
    ordering key that does not depend on parsing the timestamp strings."""
    by_id = {m["tweet_id"]: m for m in msgs}

    def depth(m):
        d, cur, seen = 0, m, set()
        while cur["parent"] in by_id and cur["tweet_id"] not in seen:
            seen.add(cur["tweet_id"])
            cur = by_id[cur["parent"]]
            d += 1
        return d

    return sorted(msgs, key=lambda m: (depth(m), int(m["tweet_id"])))


def to_record(msgs):
    """Reduce a thread to the unit the agent works on: the customer's opening
    message plus the brand's first reply to it (the historical resolution)."""
    ordered = order_thread(msgs)
    first_in = next((m for m in ordered if m["inbound"]), None)
    if first_in is None:
        return None
    idx = ordered.index(first_in)
    first_out = next((m for m in ordered[idx + 1:]
                      if not m["inbound"] and m["author_id"] == BRAND), None)
    customer_text = clean(first_in["text"])
    if len(customer_text) < 15:
        return None
    return {
        "conversation_id": first_in["tweet_id"],
        "created_at": first_in["created_at"],
        "customer_message": customer_text,
        "brand_reply": clean(first_out["text"], strip_signature=True) if first_out else None,
        "n_customer_turns": sum(1 for m in ordered if m["inbound"]),
        "n_brand_turns": sum(1 for m in ordered if not m["inbound"] and m["author_id"] == BRAND),
        # True when the opening message is the true head of the thread. Threads
        # whose head is missing from the dump start mid-conversation and are not
        # representative of an "incoming" message.
        "opens_thread": first_in["parent"] is None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(ROOT, "twcs.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "amazonhelp_threads.jsonl"))
    a = ap.parse_args()

    print(f"[1/3] building reply graph over {a.csv} ...", flush=True)
    uf, brand_roots, n = pass1(a.csv)
    print(f"      {n:,} tweets, {len(brand_roots):,} {BRAND} threads")

    print("[2/3] collecting thread text ...", flush=True)
    threads = pass2(a.csv, uf, brand_roots)
    print(f"      {sum(len(v) for v in threads.values()):,} tweets in scope")

    print("[3/3] writing records ...", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    kept = opens = 0
    with open(a.out, "w", encoding="utf-8") as f:
        for msgs in threads.values():
            rec = to_record(msgs)
            if rec:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept += 1
                opens += rec["opens_thread"]
    print(f"      wrote {kept:,} conversations ({opens:,} open their thread) -> {a.out}")


if __name__ == "__main__":
    main()
