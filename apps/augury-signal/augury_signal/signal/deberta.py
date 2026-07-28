"""Target-conditioned stance classification with DeBERTa-v3 (README enhancement 1).

The problem this solves: "Candidate X drops out" is negative *text sentiment*
and strongly *bullish* stance toward "Will Candidate Y win?". A lexicon scores
the words and gets it backwards. A model that sees the post and the claim
together can get it right.

The post and the market's target claim are encoded as a sentence pair —
`(premise=post, hypothesis=claim)` — and the model reports whether the post
supports, contradicts, or is neutral toward the claim. This is natural language
inference, and it is why an off-the-shelf NLI checkpoint works here without
task-specific fine-tuning:

    stance = P(entailment) - P(contradiction)     in [-1, 1]
    confidence = 1 - P(neutral)

That mapping has a property worth stating: a post the model finds neutral gets
a stance near 0 *and* a low confidence, so it neither pushes the signal nor
pretends to be informative.

**On the checkpoint.** Plain `microsoft/deberta-v3-base` is a pretrained
encoder with a randomly initialized classification head — it has no idea what
entailment is, and using it directly would produce confident noise. The default
here is therefore an NLI-tuned checkpoint. If you later fine-tune on labeled
prediction-market stance data, point `STANCE_MODEL_NAME` at it and bump the
version string; nothing downstream changes, and old scores stay in the database
under their own `model_version`.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from .stance import StanceScore

log = logging.getLogger(__name__)

# An NLI-tuned DeBERTa-v3 checkpoint. Zero-shot stance detection via entailment
# is a well-established use of these, and it avoids requiring a labeled corpus
# before the pipeline can run at all.
DEFAULT_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

# Label names vary across NLI checkpoints ('ENTAILMENT', 'entailment', 'LABEL_0'
# ...), so the model's own id2label mapping is inspected rather than assumed.
_ENTAIL = re.compile(r"entail", re.IGNORECASE)
_CONTRA = re.compile(r"contradict", re.IGNORECASE)
_NEUTRAL = re.compile(r"neutral", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class _LabelMap:
    """Which output index means what, resolved from the model config."""

    entailment: int
    contradiction: int
    neutral: int | None

    @classmethod
    def from_id2label(cls, id2label: dict[int, str]) -> _LabelMap:
        entailment = contradiction = neutral = None
        for index, name in id2label.items():
            index = int(index)
            if _ENTAIL.search(name):
                entailment = index
            elif _CONTRA.search(name):
                contradiction = index
            elif _NEUTRAL.search(name):
                neutral = index

        if entailment is None or contradiction is None:
            raise ValueError(
                f"cannot identify entailment/contradiction labels in {id2label!r}; "
                "STANCE_MODEL_NAME must point at an NLI checkpoint (or a stance model "
                "with equivalently named labels)"
            )
        return cls(entailment=entailment, contradiction=contradiction, neutral=neutral)


class DebertaStanceModel:
    """Zero-shot target-conditioned stance via natural language inference.

    Loading is lazy: constructing the object is cheap, and the several hundred
    megabytes of weights are only pulled when something is actually scored. That
    keeps `augury markets` and the test suite from downloading a model they will
    never use.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str = "cpu",
        batch_size: int = 16,
        max_length: int = 256,
        version_tag: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self._version_tag = version_tag
        self._tokenizer = None
        self._model = None
        self._labels: _LabelMap | None = None

    @property
    def model_version(self) -> str:
        """Identifier stored with every score.

        Includes the checkpoint name so that re-scoring with a different model
        produces a distinct series rather than silently overwriting the old one.
        """
        if self._version_tag:
            return self._version_tag
        short = self.model_name.split("/")[-1]
        return f"deberta-nli:{short}"

    def _load(self) -> None:
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "the DeBERTa stance model needs the optional NLP extras — install with "
                "`uv pip install -e '.[nlp]'` inside apps/augury-signal"
            ) from exc

        log.info("loading stance model %s on %s", self.model_name, self.device)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        model.eval()
        model.to(self.device)
        self._model = model
        self._torch = torch

        id2label = {int(k): str(v) for k, v in model.config.id2label.items()}
        self._labels = _LabelMap.from_id2label(id2label)

        if model.config.num_labels < 3:
            log.warning(
                "%s exposes %d labels; stance will be computed without a neutral class, "
                "so confidence will be less meaningful",
                self.model_name,
                model.config.num_labels,
            )

    def score(self, texts: Sequence[str], target: str) -> list[StanceScore]:
        """Score each text's stance toward `target`.

        `target` should be a declarative claim ("The Fed will cut rates in
        July"), not a question. NLI compares a premise against a *statement*;
        handing it a question makes entailment ill-defined and the outputs
        drift toward neutral.
        """
        if not texts:
            return []
        if not target or not target.strip():
            raise ValueError("target claim must not be empty")

        self._load()
        assert self._model is not None and self._tokenizer is not None and self._labels is not None

        torch = self._torch
        scores: list[StanceScore] = []

        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            encoded = self._tokenizer(
                batch,
                [target] * len(batch),
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                logits = self._model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1).cpu()

            for row in probabilities:
                p_entail = float(row[self._labels.entailment])
                p_contra = float(row[self._labels.contradiction])
                p_neutral = (
                    float(row[self._labels.neutral]) if self._labels.neutral is not None else 0.0
                )

                stance = p_entail - p_contra
                # Float error can push this a hair outside the interval, which
                # would trip the StanceScore validator.
                stance = min(1.0, max(-1.0, stance))
                confidence = min(1.0, max(0.0, 1.0 - p_neutral))

                scores.append(StanceScore(stance=stance, confidence=confidence))

        return scores

    def warm_up(self) -> None:
        """Force the weights to load now rather than on the first batch."""
        self.score(["warm up"], "This is a warm up.")
