"""Intent taxonomy for AmazonHelp, plus the escalation policy.

The eight labels were derived from the data, not invented up front: term
frequencies over 80k thread-opening messages produced seven recurring
actionable clusters, and ~21% of messages matched none of them (praise, rants,
and vague one-liners), which became the eighth. The `share` figures are the
sizes at definition time, by keyword proxy over English openers. They overlap,
because 35% of messages touch more than one topic -- which is exactly why
PRIORITY exists.

This file is also the labelling guide for the golden set: the `definition` and
`not` fields are the rules the human labeller and the LLM are both held to. The
`not` clauses were sharpened after hand-labelling 220 messages, because they are
the distinctions that actually came up.
"""

INTENTS = {
    "delivery_issue": {
        "share": "45%",
        "definition": "A shipment already in motion has gone wrong: late, lost, "
                      "marked delivered but not received, sent to the wrong address, "
                      "damaged in transit, or a complaint about the carrier's conduct.",
        "not": "If the order has not shipped yet and they only ask when it will, "
               "that is order_status. If the box arrived but its contents are "
               "wrong or faulty, that is product_or_app_issue.",
    },
    "order_status": {
        "share": "22%",
        "definition": "A question about an order that is not yet a delivery failure: "
                      "when will it ship, has it been dispatched, can I cancel or "
                      "change it, pre-order timing, order not showing in the account, "
                      "or a duplicate/extra item sent by mistake.",
        "not": "If the promised delivery date has already passed, that is delivery_issue.",
    },
    "refund_return": {
        "share": "10%",
        "definition": "Returning an item, requesting a replacement or exchange, "
                      "claiming a warranty, or chasing a refund that was promised "
                      "but has not appeared.",
        "not": "An unexpected or duplicate charge with no return involved is "
               "payment_billing.",
    },
    "payment_billing": {
        "share": "13%",
        "definition": "Money in: charged twice, charged after cancelling, card "
                      "declined, gift card or promo code not applying, cashback not "
                      "credited, price shown differs from price charged, invoices.",
        "not": "Money back out to the customer is refund_return. A charge for the "
               "Prime membership fee itself is prime_membership.",
    },
    "prime_membership": {
        "share": "15%",
        "definition": "The Prime subscription itself: signing up, cancelling, "
                      "unwanted renewal, trial conversion, or Prime benefits not "
                      "being applied -- including an unexpected or post-cancellation "
                      "charge for the membership fee.",
        "not": "'My Prime delivery is late' is delivery_issue -- Prime is only the "
               "shipping tier there, not the subject of the complaint.",
    },
    "account_access": {
        "share": "6%",
        "definition": "Getting into, securing or configuring the account itself: "
                      "login failures, password reset, two-step verification, locked "
                      "or suspended account, account compromised, changing the "
                      "registered email or phone, order visibility settings.",
        "not": "Seeing the wrong content once logged in is product_or_app_issue.",
    },
    "product_or_app_issue": {
        "share": "13%",
        "definition": "The thing itself is faulty: item arrived defective, wrong, "
                      "incomplete or counterfeit, or an Amazon digital product "
                      "misbehaves -- app or website errors, checkout failures, Prime "
                      "Video / Kindle / Alexa / Fire devices, missing episodes or "
                      "subtitles. Contents wrong, missing or faulty inside a box that "
                      "did arrive belong here, not in delivery_issue.",
        "not": "An item damaged by the courier in transit, or a box that never "
               "arrived, is delivery_issue. A request to add something to the "
               "catalogue is feedback_or_other.",
    },
    "feedback_or_other": {
        "share": "21% (no keyword match)",
        "definition": "No actionable support request: praise, thanks, general brand "
                      "ranting, jokes, catalogue or feature requests, press, seller "
                      "and partnership enquiries, or a message too vague to place in "
                      "any category above.",
        "not": "Use this as a genuine class, not as a dumping ground for messages you "
               "could resolve with one careful read.",
    },
}

LABELS = list(INTENTS)

# Applied when a message touches several topics. Ordered by which topic the
# brand must act on first: a compromised account blocks everything else, a
# failed delivery outranks a question about the same order, and money already
# taken outranks money owed back.
PRIORITY = [
    "account_access",
    "delivery_issue",
    "refund_return",
    "payment_billing",
    "prime_membership",
    "product_or_app_issue",
    "order_status",
    "feedback_or_other",
]


def resolve_conflict(labels):
    """Pick one label when several apply, using PRIORITY."""
    for lab in PRIORITY:
        if lab in labels:
            return lab
    return "feedback_or_other"


# --- Escalation policy -------------------------------------------------------
#
# "Auto-handled" here does not mean the agent resolves the case. AmazonHelp's
# own public replies almost never resolve anything on Twitter -- they
# acknowledge and hand off to a private channel. So auto-handled means: the
# drafted acknowledgement is safe to post without a human reading it first.
# Escalation means a human must look before anything goes out.

ESCALATE_RULES = [
    ("legal_or_media",
     r"\b(lawyer|solicitor|attorney|legal action|sue|suing|lawsuit|court|"
     r"ombudsman|trading standards|journalist|reporter|"
     r"consumer forum|consumer court|small claims)\b"),

    ("security_or_fraud",
     r"\b(hacked|hack|fraud|frauds|fraudulent|stolen|unauthori[sz]ed|"
     r"identity theft|scam|phishing|compromised|without permission|"
     r"didn'?t authorise|didn'?t authorize)\b"),

    ("safety_or_harm",
     r"\b(injur\w*|burn\w*|fire hazard|explod\w*|electrocut\w*|poison\w*|"
     r"choking|unsafe|dangerous|hospital|allerg\w*)\b"),

    ("vulnerable_or_distress",
     r"\b(disabled|disability|wheelchair|elderly|dementia|carer|bereave\w*|"
     r"funeral|passed away|cancer|suicid\w*|stressed|can'?t afford|"
     r"no money for)\b"),

    # Fires on >=3 stated occurrences, or on the customer saying they already
    # contacted support and it is still unresolved. Two occurrences with no
    # prior contact deliberately does NOT fire -- that is the single most common
    # delivery complaint, and escalating it would swamp the queue.
    ("repeat_contact",
     r"\b(again and again|third time|3rd time|fourth time|4th time|5th time|"
     r"umpteenth|last three|three times|3 times|many times|several times|"
     r"still no (reply|response|update|refund)|"
     r"no one (has )?(replied|responded|been able)|"
     r"spoke to \d+|speaking to \d+|called .{0,25}(support|customer|help)|"
     r"(contacted|emailed|called|chatted with) (you|them|cc|customer|support)|"
     r"customer (care|service|support)( team)? (says|said|told|gives|rude|"
     r"rep|person)|despite assurance|keep being told|not resolved|"
     r"weeks now|months now|over \d+ days|\d+ days (now|delay)|"
     r"more than a month|not yet dispatched)\b"),

    # An order number is normal here; a phone number, email address or login id
    # posted in a public tweet is a privacy problem a human must handle.
    ("pii_in_public",
     r"(\+?\d{10,}|__email__|[\w.+-]+@[\w-]+\.[a-z]{2,}|\blog ?in id\b)"),

    # Seller, vendor, affiliate and brand-abuse cases are not consumer support
    # and belong to a different team entirely.
    ("business_or_seller_dispute",
     r"\b(seller (account|funds|central)|my (brand|listing|product line)|"
     r"unauthori[sz]ed seller|affiliate|vendor|wholesale|pallet|fba|"
     r"posting privileges|my book|my title)\b"),

    ("abuse_or_severe_anger",
     r"\b(fuck|fucking|shit|bullshit|bastard|arsehole|asshole|disgusting|"
     r"disgrace|appalling|blatant lies|worst company|"
     r"never (shopping|buying|ordering) (with you )?again|"
     r"cancel(l?ing|led)? my (account|prime|prime membership|subscription))\b"),
]

# Intents that are never auto-handled regardless of wording: the standard reply
# for these would invite account or credential details into a public channel.
ALWAYS_ESCALATE_INTENTS = {"account_access"}

# Classifier confidence below this is treated as "the agent does not know".
CONFIDENCE_FLOOR = 0.55


def describe_taxonomy():
    """Render the taxonomy as prompt text / labelling instructions."""
    return "\n".join(
        f"- {name}: {spec['definition']} NOT: {spec['not']}"
        for name, spec in INTENTS.items())


if __name__ == "__main__":
    import console
    console.init()
    print(describe_taxonomy())
    print("\nPRIORITY: " + " > ".join(PRIORITY))
    print(f"\n{len(ESCALATE_RULES)} escalation rules: " +
          ", ".join(n for n, _ in ESCALATE_RULES))
