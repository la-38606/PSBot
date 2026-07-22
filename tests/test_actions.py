import pytest

from psbot.features.actions import validate_action_mask


def test_action_mask_normalizes_values() -> None:
    mask = validate_action_mask([1] + [0] * 25)
    assert mask == (True,) + (False,) * 25


def test_action_mask_rejects_wrong_size() -> None:
    with pytest.raises(ValueError, match="expected 26"):
        validate_action_mask([True])


def test_action_mask_requires_a_legal_action() -> None:
    with pytest.raises(ValueError, match="at least one"):
        validate_action_mask([False] * 26)
