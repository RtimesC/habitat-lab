"""Small, platform-independent interfaces for the navigation loop."""

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol, runtime_checkable


@dataclass
class NavigationObservation:
    """One policy input assembled from a simulator or robot observation."""

    rgb: Any
    instruction: str
    step: Optional[int] = None
    depth: Any = None
    navigation_context: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_legacy_inputs(
        cls,
        rgb_or_observation,
        instruction=None,
        step=None,
        navigation_context=None,
    ):
        """Accept the old ``rgb, instruction, ...`` policy call during migration."""
        if isinstance(rgb_or_observation, cls):
            return rgb_or_observation
        if instruction is None:
            raise ValueError(
                "instruction is required when passing a raw RGB image"
            )
        return cls(
            rgb=rgb_or_observation,
            instruction=instruction,
            step=step,
            navigation_context=navigation_context or {},
        )


@dataclass
class PolicyOutput:
    """A parsed policy decision plus the original model output for debugging."""

    action: Optional[str]
    raw_text: str
    is_valid: bool
    termination_reason: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@runtime_checkable
class NavigationPolicy(Protocol):
    """Common target-free policy interface shared by Mock and Qwen policies."""

    def predict(self, observation: NavigationObservation) -> PolicyOutput:
        ...
