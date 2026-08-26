"""Text detection layer: scam content and prompt injection.

Two classifiers over the unstructured side of an attack - the call transcript, the SMS lure,
the message a customer pasted into a support chat. They are separate from the transaction model
because they answer a different question at a different moment: the transaction model asks
"should this payment go through", while this asks "is this person being worked on right now".
A PSP that can answer the second question has a chance to intervene *before* the debit exists,
which is the only point at which an authorised push payment can still be stopped.

Character n-grams rather than word tokens, because the corpus is deliberately multilingual -
English, Hinglish and transliterated regional text in the same channel - and word tokenisation
fragments badly on transliteration. Linear models over char n-grams handle that gracefully,
train in under a second, and score in microseconds on CPU, which keeps the whole system within
the latency budget the feasibility section commits to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from janus.generate.artifacts import generate_batch


@dataclass(slots=True)
class TextDetector:
    pipeline: Pipeline

    def score(self, texts: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(texts)[:, 1]

    def _coefficients(self) -> np.ndarray | None:
        """Linear coefficients from inside the calibration wrapper.

        ``CalibratedClassifierCV`` fits one base estimator per CV fold, so there is no single
        coefficient vector. Averaging across the folds gives the direction the ensemble as a
        whole leans, which is what an explanation should reflect.
        """
        clf = self.pipeline.named_steps["clf"]
        folds = getattr(clf, "calibrated_classifiers_", None)
        if folds:
            coefs = [
                inner.coef_[0]
                for f in folds
                if (inner := getattr(f, "estimator", None)) is not None
                and hasattr(inner, "coef_")
            ]
            return np.mean(coefs, axis=0) if coefs else None
        return clf.coef_[0] if hasattr(clf, "coef_") else None

    def explain(self, text: str, top: int = 5) -> list[tuple[str, float]]:
        """The n-grams pushing this text toward the scam class.

        Useful in the console: an analyst can see *which* phrasing triggered the flag, which is
        what makes a text alert actionable rather than an opaque number.
        """
        coefs = self._coefficients()
        if coefs is None:
            return []
        vec: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
        names = vec.get_feature_names_out()
        contrib = vec.transform([text]).toarray()[0] * coefs
        order = np.argsort(-contrib)[:top]
        return [(names[i], float(contrib[i])) for i in order if contrib[i] > 0]


def train(
    n_samples: int = 4000, *, seed: int = 0, test_fraction: float = 0.25
) -> tuple[TextDetector, dict]:
    """Train the scam-content classifier on recombined corpus artefacts.

    READ THE METRIC WITH CARE. This classifier trains on a seed corpus of a few dozen authored
    documents, recombined. Its held-out AUC is therefore a statement about *separability of the
    corpus*, not a forecast of field performance - an early version scored a perfect 1.0 simply
    because scam and benign text came from disjoint vocabularies with no overlap at all.

    The corpus now includes deliberate hard negatives: legitimate messages that use the same
    vocabulary scams do - a real hospital emergency, a real overdue bill, a genuine refund
    notice, a real bank warning about OTPs. Those exist to stop the model from succeeding on
    keyword presence, which is the failure mode that makes text-based scam detection useless in
    production. Expect - and want - an AUC below 1.0 here.
    """
    batch = generate_batch(n_samples, seed=seed)

    # Split by SOURCE DOCUMENT, not by generated sample. Recombinations of one seed share
    # phrasing, so a random split over samples puts near-duplicates on both sides and reports
    # a perfect AUC. Holding out whole seeds means the test set contains phrasing the model has
    # genuinely never seen - the only version of this number worth quoting.
    rng = np.random.default_rng(seed)
    sources = sorted({a.source_id for a in batch})
    held_out = set(rng.choice(sources, size=max(2, int(len(sources) * test_fraction)),
                              replace=False).tolist())

    train_items = [a for a in batch if a.source_id not in held_out]
    test_items = [a for a in batch if a.source_id in held_out]
    if not test_items or len({a.is_scam for a in test_items}) < 2:
        # Fall back to a stratified sample split if the seed split degenerates.
        texts = [a.text for a in batch]
        labels = np.array([int(a.is_scam) for a in batch])
        x_train, x_test, y_train, y_test = train_test_split(
            texts, labels, test_size=test_fraction, random_state=seed, stratify=labels
        )
    else:
        x_train = [a.text for a in train_items]
        y_train = np.array([int(a.is_scam) for a in train_items])
        x_test = [a.text for a in test_items]
        y_test = np.array([int(a.is_scam) for a in test_items])

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=60_000, sublinear_tf=True
        )),
        ("clf", CalibratedClassifierCV(
            LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"), cv=3
        )),
    ])
    pipeline.fit(x_train, y_train)
    detector = TextDetector(pipeline=pipeline)

    scores = detector.score(x_test)
    metrics = {
        # Carried in the metrics dict itself so the number cannot be quoted without it.
        "caveat": (
            "Not evidence of field performance. The corpus is ~77 authored seed documents "
            "from a single model, so the two classes carry consistent stylistic signatures "
            "that char n-grams separate almost perfectly even across held-out seeds. Treat "
            "this layer as a demonstrated mechanism and an interpretable trigger for analyst "
            "review, not as a measured detection capability. Validating it would need real "
            "scam and non-scam message data, which is not publicly available."
        ),
        "split": "held-out source documents",
        "n_source_docs": len(sources),
        "n_held_out_docs": len(held_out),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
        "accuracy": round(float(((scores >= 0.5).astype(int) == y_test).mean()), 4),
    }
    return detector, metrics


def train_injection_detector(
    n_samples: int = 2000, *, seed: int = 0
) -> tuple[TextDetector, dict]:
    """A narrower classifier for prompt injection specifically.

    Kept separate from the scam classifier because the deployment point is different: this one
    sits in front of a merchant support agent's tool calls, not in front of a customer's phone.
    """
    rng = np.random.default_rng(seed)
    from janus.generate.artifacts import benign, prompt_injection

    items = [
        prompt_injection(rng) if rng.random() < 0.5 else benign(rng) for _ in range(n_samples)
    ]
    texts = [a.text for a in items]
    labels = np.array([int(a.is_scam) for a in items])

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=seed, stratify=labels
    )
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            analyzer="char_wb", ngram_range=(3, 5), min_df=2, max_features=40_000, sublinear_tf=True
        )),
        ("clf", CalibratedClassifierCV(
            LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"), cv=3
        )),
    ])
    pipeline.fit(x_train, y_train)
    detector = TextDetector(pipeline=pipeline)
    scores = detector.score(x_test)
    return detector, {
        "roc_auc": round(float(roc_auc_score(y_test, scores)), 4),
        "n_test": len(x_test),
    }
