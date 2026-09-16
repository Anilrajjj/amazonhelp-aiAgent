"""Retrieval over how AmazonHelp historically answered similar messages.

This is what "grounded in how the brand has resolved similar issues" means in
this system: for an incoming message we pull the nearest past customer messages
from the corpus split and hand the agent the brand's actual replies to them as
precedent.

TF-IDF over word + character n-grams rather than embeddings: the messages are
short, misspelt and full of product nouns, char n-grams absorb the typos, and
it keeps the whole pipeline reproducible in under 15 minutes on a laptop with
no model download. (See DECISIONS.md.)
"""
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.metrics.pairwise import linear_kernel

_NUM_RE = re.compile(r"\b\d[\d-]{4,}\b")


def normalise(text):
    """Order numbers and tracking ids carry no shared signal between customers
    and would only dilute the vocabulary, so they collapse to one token."""
    return _NUM_RE.sub(" <num> ", text.lower())


class PrecedentIndex:
    def __init__(self, corpus, max_features=60000):
        self.corpus = corpus
        self.vec = FeatureUnion([
            ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                     max_features=max_features, sublinear_tf=True)),
            ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                     max_features=max_features, sublinear_tf=True)),
        ])
        # a list, not a generator: FeatureUnion iterates the input once per
        # transformer and the second one would otherwise see an exhausted stream
        docs = [normalise(r["customer_message"]) for r in corpus]
        self.matrix = self.vec.fit_transform(docs)

    def search(self, message, k=5):
        """Return the k most similar past conversations, best first."""
        q = self.vec.transform([normalise(message)])
        sims = linear_kernel(q, self.matrix).ravel()
        top = sims.argsort()[::-1][:k]
        return [{
            "score": float(sims[i]),
            "customer_message": self.corpus[i]["customer_message"],
            "brand_reply": self.corpus[i]["brand_reply"],
            "conversation_id": self.corpus[i]["conversation_id"],
        } for i in top]


if __name__ == "__main__":
    import sys
    import console; console.init()
    from dataset import load_split
    corpus, pool = load_split()
    print(f"indexing {len(corpus):,} past conversations ...")
    idx = PrecedentIndex(corpus)
    q = sys.argv[1] if len(sys.argv) > 1 else \
        "my parcel says delivered but it never arrived, this is the second time"
    print(f"\nquery: {q}\n")
    for h in idx.search(q):
        print(f"[{h['score']:.3f}] C: {h['customer_message'][:110]}")
        print(f"         A: {h['brand_reply'][:110]}\n")
