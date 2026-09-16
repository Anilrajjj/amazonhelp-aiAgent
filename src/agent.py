"""The AmazonHelp support agent: classify -> draft -> route.

Three decisions per incoming message:

1. classify  -- one of the eight intents in intents.py, with a confidence.
2. draft     -- a reply grounded in how AmazonHelp actually answered the most
                similar past messages (retrieved from the corpus split).
3. route     -- auto-handle or escalate, with a stated reason.

Classification and drafting are one LLM call, not two. The precedent replies
are the most expensive part of the prompt and both steps need them; splitting
the call doubled cost for no measurable accuracy gain.

Routing is deliberately NOT left to the LLM alone. The hard rules in
intents.ESCALATE_RULES fire first and cannot be overridden, because the cost of
auto-replying to a fraud report or a legal threat is not symmetric with the cost
of escalating a routine one. The LLM only gets to escalate cases the rules
missed; it never gets to de-escalate.
"""
import json, re

import llm
from intents import (ALWAYS_ESCALATE_INTENTS, CONFIDENCE_FLOOR, ESCALATE_RULES,
                     LABELS, describe_taxonomy)
from dataset import is_english

SYSTEM = (
    "You are a customer-support triage assistant for Amazon's Twitter support "
    "account (@AmazonHelp). You classify an incoming public tweet and draft the "
    "brand's public reply. You are precise and you never invent facts."
)

PROMPT = """Classify the customer message below into exactly one intent, then draft AmazonHelp's public reply.

INTENTS:
{taxonomy}

If several intents apply, pick the one the brand must act on first, using this order:
{priority}

HOW THIS BRAND ACTUALLY REPLIES -- the most similar past messages and the real replies AmazonHelp sent:
{precedents}

RULES FOR THE DRAFT:
- Match the voice, length and structure of the precedent replies above. They are one or two sentences.
- AmazonHelp cannot see orders or accounts from Twitter. Never state a delivery date, refund amount, order status or account fact. Never promise a specific outcome or timeframe.
- Where the precedents hand off to a private channel, do the same, and write the placeholder <link> exactly -- never invent a URL.
- Acknowledge the specific problem the customer described, not a generic one.
- No hashtags, no emoji, no agent sign-off initials.

CUSTOMER MESSAGE:
\"\"\"{message}\"\"\"

Return ONLY a JSON object:
{{"intent": "<one intent name>",
  "confidence": <0.0-1.0, how sure you are of the intent>,
  "reply": "<the draft reply>",
  "needs_human": <true if a human must review before this is sent>,
  "human_reason": "<short reason, or null>"}}"""


def rule_escalations(message):
    """Hard escalation triggers. Returns the list of rule names that fired."""
    text = message.lower()
    return [name for name, pat in ESCALATE_RULES if re.search(pat, text)]


def format_precedents(hits):
    out = []
    for i, h in enumerate(hits, 1):
        out.append(f"{i}. customer: {h['customer_message']}\n   AmazonHelp: {h['brand_reply']}")
    return "\n".join(out)


def decide_route(intent, confidence, fired_rules, llm_needs_human, llm_reason, message):
    """Combine the hard rules, the intent policy, the confidence floor and the
    model's own judgement into one route plus a stated reason.

    Order matters: anything a hard rule caught is escalated whatever the model
    thinks, and the model can only add escalations, never remove them."""
    if not is_english(message):
        return "escalate", "non_english"
    if fired_rules:
        return "escalate", fired_rules[0]
    if intent in ALWAYS_ESCALATE_INTENTS:
        return "escalate", "account_security_policy"
    if confidence < CONFIDENCE_FLOOR:
        return "escalate", "low_confidence"
    if llm_needs_human:
        return "escalate", llm_reason or "model_flagged"
    return "auto", None


def handle(message, index, k=5):
    """Run the full agent on one incoming message."""
    hits = index.search(message, k=k)
    prompt = PROMPT.format(
        taxonomy=describe_taxonomy(),
        priority=" > ".join(__import__("intents").PRIORITY),
        precedents=format_precedents(hits),
        message=message.replace('"""', "'''"),
    )

    try:
        out = llm.chat_json(prompt, system=SYSTEM, max_tokens=900)
    except Exception as e:
        # A dead API must not look like a confident prediction.
        return {
            "intent": None, "confidence": 0.0, "reply": None,
            "route": "escalate", "escalation_reason": "agent_error",
            "error": f"{type(e).__name__}: {e}",
            "precedent_ids": [h["conversation_id"] for h in hits],
        }

    intent = out.get("intent")
    if intent not in LABELS:
        intent, conf = "feedback_or_other", 0.0
    else:
        conf = float(out.get("confidence") or 0.0)

    fired = rule_escalations(message)
    route, reason = decide_route(
        intent, conf, fired, bool(out.get("needs_human")),
        out.get("human_reason"), message)

    return {
        "intent": intent,
        "confidence": conf,
        "reply": (out.get("reply") or "").strip(),
        "route": route,
        "escalation_reason": reason,
        "rules_fired": fired,
        "precedent_ids": [h["conversation_id"] for h in hits],
        "top_precedent_score": hits[0]["score"] if hits else 0.0,
    }


if __name__ == "__main__":
    import sys
    import console
    console.init()
    from dataset import load_split
    from retrieval import PrecedentIndex

    if not llm.is_available():
        raise SystemExit(
            "No LLM configured. Set one of OPENROUTER_API_KEY / GROQ_API_KEY / "
            "NVIDIA_API_KEY (and optionally LLM_PROVIDER, LLM_MODEL).")

    corpus, _ = load_split()
    print(f"indexing {len(corpus):,} past conversations ...")
    index = PrecedentIndex(corpus)

    msg = " ".join(sys.argv[1:]) or \
        "my parcel says delivered but it never arrived and no one has replied to me"
    print(f"\nprovider={llm.active_provider()} model={llm.model_name()}")
    print(f"\ncustomer: {msg}\n")
    print(json.dumps(handle(msg, index), indent=2, ensure_ascii=False))
