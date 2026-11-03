from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoherenceClipConfig:
    train_coherence_filters: bool = False
    threshold_bias_axis: float | None = None
    max_iterations: float | None = None
    exclude_llrs_smaller_than: float | None = None
    train_clip_denominators: bool = False
    clip_denominator_percentage: float | None = None

    def _fmt(self, x):
        if x is None:
            return "None"
        if x == np.inf:
            return "inf"
        if x == -np.inf:
            return "-inf"
        return str(x)

    def key(self) -> str:
        parts = []

        if self.train_coherence_filters:
            parts.append(f"threshold_bias_axis_{self._fmt(self.threshold_bias_axis)}")
            parts.append(f"max_iterations_{self._fmt(self.max_iterations)}")
            parts.append(f"exclude_llrs_smaller_than_{self._fmt(self.exclude_llrs_smaller_than)}")

        # Clip only if active
        if self.train_clip_denominators is not None:
            parts.append(f"clip_denominator_{self._fmt(self.clip_denominator_percentage)}")

        return "_".join(parts) if parts else "default"
