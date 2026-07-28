"""Stance scoring: how much a post supports a market's target claim.

The interface is deliberately narrow — `score(texts, target) -> [StanceScore]` —
so the DeBERTa model added in Stage 4 drops in behind it without anything
downstream changing.

Stance is not sentiment, and the distinction is the reason this abstraction
exists. "Candidate X drops out" is negative *text sentiment* and strongly
bullish *stance* toward "Will Candidate Y win?". A lexicon scores the words; a
target-conditioned model scores the claim. VADER is kept anyway as a baseline
and as the second opinion in the ensemble, because knowing when the two
disagree is itself a useful data-quality signal.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class StanceScore:
    """One model's read on one post.

    `stance` in [-1, 1] runs bearish to bullish on the target claim.
    `confidence` in [0, 1] is how strongly the model holds that view.
    """

    stance: float
    confidence: float

    def __post_init__(self) -> None:
        if not -1.0 <= self.stance <= 1.0:
            raise ValueError(f"stance {self.stance} outside [-1, 1]")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")


@runtime_checkable
class StanceModel(Protocol):
    """Anything that can score posts against a target claim."""

    @property
    def model_version(self) -> str:
        """Stable identifier recorded with every score.

        Scores from different versions are different series and must never be
        concatenated — which is why `stances` is keyed on this in the database.
        """
        ...

    def score(self, texts: Sequence[str], target: str) -> list[StanceScore]:
        """Score a batch of texts against one target claim."""
        ...


class VaderStanceModel:
    """Lexicon baseline: VADER's compound sentiment, used as a stance proxy.

    Honest about what it is. VADER ignores `target` entirely — it cannot, by
    construction, tell that bad news for one side of a market is good news for
    the other. It is here to be the cheap baseline the real model has to beat,
    and to supply the second opinion in `EnsembleStanceModel`. Do not ship
    results computed from this alone as evidence about stance.

    Confidence is the magnitude of the compound score: VADER exposes no
    uncertainty of its own, and |compound| is at least monotone in how strongly
    the lexicon fired.
    """

    def __init__(self) -> None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        self._analyzer = SentimentIntensityAnalyzer()

    @property
    def model_version(self) -> str:
        return "vader-3.3.2"

    def score(self, texts: Sequence[str], target: str) -> list[StanceScore]:  # noqa: ARG002
        scores: list[StanceScore] = []
        for text in texts:
            compound = self._analyzer.polarity_scores(text)["compound"]
            scores.append(StanceScore(stance=float(compound), confidence=abs(float(compound))))
        return scores


class EnsembleStanceModel:
    """Runs two models and reports the primary's score plus their disagreement.

    The disagreement is the product: |stance_primary - stance_secondary| is a
    per-post data-quality flag. A post the target-conditioned model reads as
    bullish and the lexicon reads as bearish is usually one of the cases where
    stance and sentiment genuinely diverge — which is either the model working
    exactly as intended, or the post being ambiguous enough to distrust. Either
    way it is worth being able to filter on afterwards.
    """

    def __init__(self, primary: StanceModel, secondary: StanceModel) -> None:
        self.primary = primary
        self.secondary = secondary

    @property
    def model_version(self) -> str:
        return f"{self.primary.model_version}+{self.secondary.model_version}"

    def score(self, texts: Sequence[str], target: str) -> list[StanceScore]:
        return self.primary.score(texts, target)

    def score_with_disagreement(
        self, texts: Sequence[str], target: str
    ) -> list[tuple[StanceScore, float]]:
        """Return each primary score paired with its disagreement value."""
        primary = self.primary.score(texts, target)
        secondary = self.secondary.score(texts, target)
        return [
            (p, abs(p.stance - s.stance)) for p, s in zip(primary, secondary, strict=True)
        ]


def build_stance_model(name: str, **kwargs: object) -> StanceModel:
    """Construct a stance model by name, per the STANCE_MODEL setting.

    'deberta' is wired up in Stage 4; until then it raises rather than silently
    falling back to VADER, because a run that believes it used a
    target-conditioned model but did not would produce quietly wrong results.
    """
    key = name.strip().lower()

    if key == "vader":
        return VaderStanceModel()

    if key == "deberta":
        from .deberta import DebertaStanceModel

        return DebertaStanceModel(**kwargs)  # type: ignore[arg-type]

    raise ValueError(f"unknown stance model {name!r}; expected 'vader' or 'deberta'")
