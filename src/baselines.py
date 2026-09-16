"""Two baselines the agent has to beat.

trivial -- predicts the majority intent for everything, never escalates, and
    sends one fixed canned reply. This exists to show what the headline numbers
    look like when the system does nothing at all. On a skewed support queue a
    trivial baseline is embarrassingly hard to beat on accuracy, which is the
    point.

simple -- no LLM anywhere:
    intent : TF-IDF + logistic regression, trained on weak keyword labels over
             the retrieval corpus (never on the golden set).
    reply  : the historical AmazonHelp reply to the single nearest past message,
             copied verbatim. This is a genuinely strong reply baseline, because
             the brand's real replies are templated -- if a generated reply
             cannot beat copy-the-nearest-neighbour, generation is not earning
             its cost.
    route  : the hard escalation rules only.

Both are deterministic and need no API key.
"""
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from intents import ESCALATE_RULES, resolve_conflict
from dataset import is_english

MAJORITY_INTENT = "delivery_issue"

CANNED_REPLY = ("I'm sorry for the trouble. Please contact us here: <link> "
                "and we'll be happy to help.")

# Weak-supervision patterns. Same vocabulary as the sampling prior in
# build_golden.py, reused here on purpose: the simple baseline is allowed the
# same cheap signal the sampler had, so the comparison is about the model, not
# about who got better keywords.
WEAK_PATTERNS = {
    "delivery_issue": r"deliver|package|parcel|courier|carrier|shipped|shipping|arriv|tracking|lost|late|missing",
    "order_status": r"\border\b|dispatch|when will|status|cancel my order|change my order|preorder|pre-order",
    "refund_return": r"refund|return|replacement|exchange|money back|reimburs",
    "payment_billing": r"charg|payment|card|billed|billing|invoice|price|paid|\bpay\b|gift card|discount|coupon|cashback",
    "prime_membership": r"prime|membership|subscription|subscribe|renew",
    "account_access": r"account|login|log in|sign in|password|hacked|locked|two-step|verification|email address",
    "product_or_app_issue": r"\bapp\b|website|site|error|crash|kindle|echo|alexa|fire ?stick|prime video|episode|broken|defective|damaged|fake|counterfeit|wrong item|not work",
}


def weak_label(text):
    t = text.lower()
    hits = [k for k, p in WEAK_PATTERNS.items() if re.search(p, t)]
    return resolve_conflict(hits) if hits else "feedback_or_other"


def rule_route(message):
    """Escalation by hard rules only -- no intent policy, no confidence."""
    if not is_english(message):
        return "escalate", "non_english"
    for name, pat in ESCALATE_RULES:
        if re.search(pat, message.lower()):
            return "escalate", name
    return "auto", None


class TrivialBaseline:
    name = "trivial (majority intent + canned reply + never escalate)"

    def handle(self, message, **_):
        return {"intent": MAJORITY_INTENT, "confidence": 1.0,
                "reply": CANNED_REPLY, "route": "auto", "escalation_reason": None}


class SimpleBaseline:
    name = "simple (TF-IDF logistic regression + nearest-neighbour reply + rules)"

    def __init__(self, corpus, index):
        self.index = index
        texts = [r["customer_message"] for r in corpus]
        labels = [weak_label(t) for t in texts]
        self.clf = make_pipeline(
            TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True,
                            max_features=60000),
            LogisticRegression(max_iter=1000, C=4.0, class_weight="balanced"),
        )
        self.clf.fit(texts, labels)

    def handle(self, message, **_):
        intent = self.clf.predict([message])[0]
        conf = float(self.clf.predict_proba([message]).max())
        hits = self.index.search(message, k=1)
        route, reason = rule_route(message)
        return {"intent": intent, "confidence": conf,
                "reply": hits[0]["brand_reply"] if hits else CANNED_REPLY,
                "route": route, "escalation_reason": reason,
                "precedent_ids": [h["conversation_id"] for h in hits],
                "top_precedent_score": hits[0]["score"] if hits else 0.0}
