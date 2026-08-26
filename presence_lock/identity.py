"""Enrolled face templates: storage, matching, and quality statistics.

A template is a set of L2-normalised embeddings — several per head pose — for a
single person. Matching takes the *maximum* cosine similarity across the set
rather than the distance to a centroid, so an unusual pose is scored against
the sample that actually looks like it instead of against an average face.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import IDENTITY_DIR

log = logging.getLogger(__name__)

_SUFFIX = ".npz"


@dataclass
class MatchResult:
    name: str
    similarity: float
    pose: str
    is_match: bool
    is_stranger: bool


class Identity:
    """One enrolled person."""

    def __init__(
        self,
        name: str,
        embeddings: np.ndarray,
        poses: list[str],
        backend: str,
        threshold: float,
        created: float | None = None,
        meta: dict | None = None,
    ) -> None:
        self.name = name
        array = np.asarray(embeddings, dtype=np.float32)
        self.embeddings = array.reshape(-1, array.shape[-1])
        self.poses = list(poses)
        self.backend = backend
        self.threshold = float(threshold)
        self.created = float(created if created is not None else time.time())
        self.meta = dict(meta or {})

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.embeddings)

    @property
    def path(self) -> Path:
        return IDENTITY_DIR / f"{self.name}{_SUFFIX}"

    @property
    def centroid(self) -> np.ndarray:
        centre = self.embeddings.mean(axis=0)
        norm = float(np.linalg.norm(centre))
        return centre if norm == 0 else centre / norm

    # ------------------------------------------------------------------
    def similarity(self, embedding: np.ndarray) -> tuple[float, int]:
        """Best cosine similarity against the template, plus its sample index."""
        if embedding is None or len(self.embeddings) == 0:
            return -1.0, -1
        scores = self.embeddings @ np.asarray(embedding, dtype=np.float32).ravel()
        index = int(np.argmax(scores))
        return float(scores[index]), index

    def add(self, embedding: np.ndarray, pose: str, dedup_threshold: float = 0.995) -> bool:
        """Append a sample unless a near-identical one is already stored."""
        embedding = np.asarray(embedding, dtype=np.float32).ravel()
        if len(self.embeddings):
            best, _ = self.similarity(embedding)
            if best >= dedup_threshold:
                return False
            self.embeddings = np.vstack([self.embeddings, embedding[None, :]])
        else:
            self.embeddings = embedding[None, :]
        self.poses.append(pose)
        return True

    def intra_stats(self) -> dict[str, float]:
        """Pairwise similarity spread inside the template.

        A low 5th percentile means the enrolled poses disagree with each other,
        which is exactly what a strict threshold would reject in daily use.
        """
        count = len(self.embeddings)
        if count < 2:
            return {"min": 1.0, "p05": 1.0, "mean": 1.0}
        gram = self.embeddings @ self.embeddings.T
        upper = gram[np.triu_indices(count, k=1)]
        return {
            "min": float(upper.min()),
            "p05": float(np.percentile(upper, 5)),
            "mean": float(upper.mean()),
        }

    # ------------------------------------------------------------------
    def save(self) -> Path:
        IDENTITY_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            self.path,
            embeddings=self.embeddings.astype(np.float32),
            poses=np.array(self.poses, dtype=object),
            backend=self.backend,
            threshold=self.threshold,
            created=self.created,
            meta=json.dumps(self.meta),
        )
        return self.path

    @classmethod
    def load(cls, path: Path) -> "Identity":
        with np.load(path, allow_pickle=True) as data:
            meta_raw = data["meta"].item() if "meta" in data else "{}"
            return cls(
                name=path.stem,
                embeddings=data["embeddings"],
                poses=[str(p) for p in data["poses"].tolist()],
                backend=str(data["backend"].item()),
                threshold=float(data["threshold"].item()),
                created=float(data["created"].item()),
                meta=json.loads(meta_raw) if meta_raw else {},
            )

    @classmethod
    def load_by_name(cls, name: str) -> "Identity":
        path = IDENTITY_DIR / f"{name}{_SUFFIX}"
        if not path.exists():
            raise FileNotFoundError(f"no enrolled identity named '{name}' in {IDENTITY_DIR}")
        return cls.load(path)


# ----------------------------------------------------------------------
# Collection
# ----------------------------------------------------------------------


def list_identities() -> list[str]:
    if not IDENTITY_DIR.exists():
        return []
    return sorted(p.stem for p in IDENTITY_DIR.glob(f"*{_SUFFIX}"))


class IdentityGallery:
    """The set of identities allowed to hold the session open."""

    def __init__(
        self,
        identities: list[Identity],
        threshold: float = 0.0,
        margin: float = 0.08,
    ) -> None:
        if not identities:
            raise ValueError("no enrolled identities — run `python plock.py enroll` first")
        self.identities = identities
        self.threshold = float(threshold) if threshold > 0 else max(i.threshold for i in identities)
        self.margin = float(margin)

    @classmethod
    def load(
        cls,
        name: str = "",
        backend_name: str = "",
        threshold: float = 0.0,
        margin: float = 0.08,
    ) -> "IdentityGallery":
        names = [name] if name else list_identities()
        if not names:
            raise FileNotFoundError(
                "no enrolled identities — run `python plock.py enroll --name <you>` first"
            )
        loaded = [Identity.load_by_name(n) for n in names]
        if backend_name:
            usable = [i for i in loaded if i.backend == backend_name]
            skipped = [i.name for i in loaded if i.backend != backend_name]
            if skipped:
                log.warning(
                    "Ignoring identities enrolled with another backend: %s "
                    "(re-enrol them with the current backend '%s')",
                    ", ".join(skipped),
                    backend_name,
                )
            loaded = usable
        if not loaded:
            raise RuntimeError(
                f"no identity was enrolled with backend '{backend_name}'; re-run enrolment"
            )
        return cls(loaded, threshold=threshold, margin=margin)

    @property
    def names(self) -> list[str]:
        return [i.name for i in self.identities]

    def match(self, embedding: np.ndarray | None) -> MatchResult:
        if embedding is None:
            return MatchResult("", -1.0, "", False, False)

        best_name, best_sim, best_pose = "", -1.0, ""
        for identity in self.identities:
            sim, index = identity.similarity(embedding)
            if sim > best_sim:
                best_name, best_sim = identity.name, sim
                best_pose = identity.poses[index] if 0 <= index < len(identity.poses) else ""

        return MatchResult(
            name=best_name,
            similarity=best_sim,
            pose=best_pose,
            is_match=best_sim >= self.threshold,
            # The gap between "not confident enough" and "definitely someone
            # else" keeps an awkwardly-posed owner out of the stranger bucket.
            is_stranger=best_sim < (self.threshold - self.margin),
        )
