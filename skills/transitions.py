"""Deterministic stand/walk transition schedules and acceptance summaries."""

import math
from dataclasses import dataclass
from typing import Literal

SkillName = Literal["stand", "walk"]


@dataclass(frozen=True)
class TransitionPhase:
    """One hold or linear motor-target blend in a connected rollout."""

    name: str
    duration_seconds: float
    source: SkillName
    target: SkillName | None = None

    @property
    def is_blend(self) -> bool:
        """Return whether this phase mixes two skill controllers."""
        return self.target is not None


@dataclass(frozen=True)
class TransitionDirective:
    """Skill selection and blend fraction for one control step."""

    phase_name: str
    source: SkillName
    target: SkillName | None
    blend: float


DEFAULT_TRANSITION_PHASES = (
    TransitionPhase("stand_initial", 2.0, "stand"),
    TransitionPhase("start_1", 0.5, "stand", "walk"),
    TransitionPhase("walk_1", 4.0, "walk"),
    TransitionPhase("stop_1", 0.5, "walk", "stand"),
    TransitionPhase("stand_middle", 2.0, "stand"),
    TransitionPhase("start_2", 0.5, "stand", "walk"),
    TransitionPhase("walk_2", 4.0, "walk"),
    TransitionPhase("stop_2", 0.5, "walk", "stand"),
    TransitionPhase("stand_final", 10.0, "stand"),
)


def build_transition_directives(
    dt: float,
    phases: tuple[TransitionPhase, ...] = DEFAULT_TRANSITION_PHASES,
) -> list[TransitionDirective]:
    """Expand a phase schedule to one directive per control step.

    A blend starts strictly above zero and ends at one, so its last step is
    controlled entirely by the incoming skill before the following hold.
    """
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    if not phases:
        raise ValueError("At least one transition phase is required")

    directives: list[TransitionDirective] = []
    previous_skill: SkillName | None = None
    for phase in phases:
        if (
            not phase.name
            or not math.isfinite(phase.duration_seconds)
            or phase.duration_seconds <= 0
        ):
            raise ValueError("Transition phases need a name and positive duration")
        if previous_skill is not None and phase.source != previous_skill:
            raise ValueError(
                f"Phase {phase.name!r} starts with {phase.source!r}, "
                f"expected {previous_skill!r}"
            )
        if phase.target == phase.source:
            raise ValueError("A blend target must differ from its source")
        steps = round(phase.duration_seconds / dt)
        if steps < 1 or not math.isclose(
            steps * dt, phase.duration_seconds, abs_tol=1e-9
        ):
            raise ValueError(
                f"Phase {phase.name!r} duration must be an integer multiple of dt"
            )
        for offset in range(steps):
            blend = (offset + 1) / steps if phase.is_blend else 0.0
            directives.append(
                TransitionDirective(
                    phase_name=phase.name,
                    source=phase.source,
                    target=phase.target,
                    blend=blend,
                )
            )
        previous_skill = phase.target or phase.source
    return directives


def transition_boundaries(
    directives: list[TransitionDirective],
) -> list[tuple[int, SkillName]]:
    """Return step indices where an incoming controller must be activated."""
    boundaries: list[tuple[int, SkillName]] = [(0, directives[0].source)]
    for index, directive in enumerate(directives):
        if directive.target is None:
            continue
        if index == 0 or directives[index - 1].phase_name != directive.phase_name:
            boundaries.append((index, directive.target))
    return boundaries
