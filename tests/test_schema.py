from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from psbot.data.schema import EpisodeMetadata, EpisodeSource, Transition


def test_episode_metadata_accepts_timezone_aware_timestamp() -> None:
    metadata = EpisodeMetadata(
        battle_id="battle-1",
        source=EpisodeSource.LOCAL,
        timestamp=datetime(2026, 7, 21, tzinfo=UTC),
        winner="p1",
        simulator_commit="f2fe71c",
    )
    assert metadata.battle_format == "gen9randombattle"


def test_episode_metadata_rejects_naive_timestamp() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        EpisodeMetadata(
            battle_id="battle-1",
            source=EpisodeSource.LOCAL,
            timestamp=datetime(2026, 7, 21),
            winner="p1",
            simulator_commit="f2fe71c",
        )


def test_transition_requires_chosen_action_to_be_legal() -> None:
    with pytest.raises(ValidationError, match="chosen action is illegal"):
        Transition(
            battle_id="battle-1",
            pov="p1",
            turn=1,
            observation={},
            chosen_action=3,
            legal_action_mask=(True,) + (False,) * 25,
            reward=0.0,
            terminal=False,
            outcome=0,
        )
