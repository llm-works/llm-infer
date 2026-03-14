"""Progress tracking for model loading."""

from typing import Any

from appinfra.log import Logger
from appinfra.time import ETA, delta_str, since, start

_PHASE_LABELS = {
    "tokenizer": ("tokenizer", None),
    "weights:init": ("weights", "initialized"),
    "weights:alloc": ("weights", "allocated"),
    "weights:stream": ("weights", "loaded"),
    "kv_cache": ("kv_cache", None),
}

_PHASE_ACTIONS = {
    "weights:init": ("initializing", "initialized"),
    "weights:alloc": ("allocating", "allocated"),
    "weights:stream": ("loading", "loaded"),
}


class ProgressTracker:
    """Tracks loading progress with timing and ETA for each phase."""

    def __init__(self, lg: Logger) -> None:
        self._lg = lg
        self._start_times: dict[str, float] = {}
        self._last_logged: dict[str, int] = {}
        self._etas: dict[str, ETA | None] = {}

    def __call__(self, phase: str, current: int, total: int) -> None:
        label, progress_field = _PHASE_LABELS.get(phase, (phase, None))
        action_ing, action_ed = _PHASE_ACTIONS.get(phase, ("loading", "loaded"))

        # Ensure phase is initialized before progress/completion
        if phase not in self._start_times:
            self._on_phase_start(phase, total, action_ing, label)

        if current >= total:
            self._on_phase_complete(phase, action_ed, label)
        elif current > 0:
            self._on_phase_progress(
                phase, current, total, action_ing, label, progress_field
            )

    def _on_phase_start(
        self, phase: str, total: int, action_ing: str, label: str
    ) -> None:
        self._start_times[phase] = start()
        self._last_logged[phase] = 0
        self._etas[phase] = ETA(total=total) if total > 1 else None
        self._lg.debug(f"{action_ing} {label}...")

    def _build_progress_extra(
        self, phase: str, current: int, total: int, progress_field: str | None
    ) -> dict[str, Any]:
        """Build extra dict for progress logging."""
        extra: dict[str, Any] = {
            "after": since(self._start_times[phase]),
            "total": total,
            "progress": f"{(current * 100) // total}%",
        }
        if progress_field is not None:
            extra[progress_field] = current
        if eta_obj := self._etas.get(phase):
            eta_obj.update(current)
            if (remaining := eta_obj.remaining_secs()) is not None:
                extra["eta"] = delta_str(remaining)
        return extra

    def _on_phase_progress(
        self,
        phase: str,
        current: int,
        total: int,
        action_ing: str,
        label: str,
        progress_field: str | None,
    ) -> None:
        step = max(1, total // 10)
        if current - self._last_logged.get(phase, 0) < step:
            return
        self._lg.debug(
            f"{action_ing} {label}...",
            extra=self._build_progress_extra(phase, current, total, progress_field),
        )
        self._last_logged[phase] = current

    def _on_phase_complete(self, phase: str, action_ed: str, label: str) -> None:
        elapsed = since(self._start_times[phase])
        self._lg.info(f"{label} {action_ed}", extra={"after": elapsed})
