"""Interactive tester: type a customer message, see what the agent does.

Builds the precedent index once (~1 minute) and then answers instantly, which is
the difference between trying three messages and trying thirty.

    python src/try_agent.py                 # interactive
    python src/try_agent.py --demo          # run a fixed set of probe messages
    python src/try_agent.py --show-precedent  # also print what it retrieved

Commands inside the session:  :q quit   :p toggle precedent   :h help
"""
import argparse, json, sys

import console
import llm
from dataset import load_split
from retrieval import PrecedentIndex
import agent as agent_mod

# Probe messages chosen to hit a different branch of the routing logic each time,
# so a reviewer can see the policy work rather than just the happy path.
DEMO = [
    ("plain delivery complaint",
     "my parcel was supposed to arrive yesterday and it still hasn't turned up"),
    ("repeat contact -> escalate",
     "this is the third time my order has been late and no one has replied to me"),
    ("fraud -> escalate",
     "there are unauthorised transactions on my saved card, someone has used it"),
    ("account -> always escalate by policy",
     "i can't log into my account, it says my password is wrong"),
    ("praise -> no action needed",
     "just got my order in 6 hours, honestly incredible service"),
    ("catalogue request",
     "will you ever add more Studio Ghibli films to Prime Video?"),
    ("abuse -> escalate",
     "this is absolute bullshit, i'm cancelling my prime membership today"),
    ("non-English -> escalate",
     "mi pedido no ha llegado y necesito ayuda urgente por favor"),
]


def show(msg, out, index=None, show_precedent=False):
    route = out.get("route")
    mark = "AUTO " if route == "auto" else "ESCALATE"
    print(f"\n  customer : {msg}")
    print(f"  intent   : {out.get('intent')}  (confidence {out.get('confidence')})")
    print(f"  route    : {mark}", end="")
    if out.get("escalation_reason"):
        print(f"  <- {out['escalation_reason']}")
    else:
        print()
    if out.get("rules_fired"):
        print(f"  rules    : {', '.join(out['rules_fired'])}")
    print(f"  reply    : {out.get('reply') or '(none)'}")
    if out.get("error"):
        print(f"  error    : {out['error']}")
    if show_precedent and index is not None:
        print("  grounded on:")
        for h in index.search(msg, k=3):
            print(f"    [{h['score']:.2f}] C: {h['customer_message'][:80]}")
            print(f"           A: {(h['brand_reply'] or '')[:80]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true",
                    help="run a fixed set of probe messages and exit")
    ap.add_argument("--show-precedent", action="store_true",
                    help="also print the retrieved historical replies")
    a = ap.parse_args()
    console.init()

    if not llm.is_available():
        raise SystemExit(
            "No LLM configured. Create .env in the repo root with:\n"
            "  LLM_PROVIDER=openrouter\n  OPENROUTER_API_KEY=sk-or-...\n"
            "(see .env.example)")

    corpus, _ = load_split()
    print(f"indexing {len(corpus):,} past AmazonHelp conversations ...", flush=True)
    index = PrecedentIndex(corpus)
    print(f"ready.  provider={llm.active_provider()}  model={llm.model_name()}")

    if a.demo:
        for label, msg in DEMO:
            print(f"\n{'-' * 70}\n# {label}")
            show(msg, agent_mod.handle(msg, index), index, a.show_precedent)
        return

    show_prec = a.show_precedent
    print("\nType a customer message.  :q quit   :p toggle precedent   :h help")
    while True:
        try:
            msg = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not msg:
            continue
        if msg in (":q", ":quit", "exit"):
            return
        if msg == ":p":
            show_prec = not show_prec
            print(f"  precedent display {'on' if show_prec else 'off'}")
            continue
        if msg in (":h", ":help"):
            print("  :q quit   :p toggle precedent display   anything else = a message")
            continue
        show(msg, agent_mod.handle(msg, index), index, show_prec)


if __name__ == "__main__":
    main()
