"""Action-space validation helpers."""

from collections.abc import Sequence

from psbot.constants import ACTION_SPACE_SIZE


def validate_action_mask(mask: Sequence[bool | int]) -> tuple[bool, ...]:
    """Return a normalized legal-action mask or raise on a broken contract."""

    if len(mask) != ACTION_SPACE_SIZE:
        raise ValueError(f"expected {ACTION_SPACE_SIZE} actions, received {len(mask)}")
    normalized = tuple(bool(value) for value in mask)
    if not any(normalized):
        raise ValueError("an action mask must contain at least one legal action")
    return normalized
