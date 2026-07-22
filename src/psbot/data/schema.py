"""Schema v1 shared by human replay and self-play datasets."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from psbot.constants import BATTLE_FORMAT, SCHEMA_VERSION
from psbot.features.actions import validate_action_mask


class EpisodeSource(StrEnum):
    HUMAN_REPLAY = "human-replay"
    SELF_PLAY = "self-play"
    LOCAL = "local"
    LADDER = "ladder"


class EpisodeMetadata(BaseModel):
    """Irrecoverable battle-level metadata frozen before collection starts."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    battle_id: str = Field(min_length=1)
    source: EpisodeSource
    battle_format: str = BATTLE_FORMAT
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    winner: Literal["p1", "p2", "tie"]
    p1_rating: float | None = None
    p2_rating: float | None = None
    simulator_commit: str = Field(min_length=7)
    seed: str | None = None
    p1_hash: str | None = None
    p2_hash: str | None = None
    protocol_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @field_validator("timestamp")
    @classmethod
    def timestamp_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp must include a timezone")
        return value


class Transition(BaseModel):
    """One player decision from a strictly first-person point of view."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    battle_id: str = Field(min_length=1)
    pov: Literal["p1", "p2"]
    turn: int = Field(ge=0)
    observation: dict[str, Any]
    chosen_action: int = Field(ge=0, le=25)
    legal_action_mask: tuple[bool, ...]
    reward: float
    terminal: bool
    outcome: Literal[-1, 0, 1]

    @field_validator("legal_action_mask", mode="before")
    @classmethod
    def normalize_mask(cls, value: Any) -> tuple[bool, ...]:
        return validate_action_mask(value)

    @model_validator(mode="after")
    def chosen_action_must_be_legal(self) -> Transition:
        if not self.legal_action_mask[self.chosen_action]:
            raise ValueError("chosen action is illegal under the recorded mask")
        return self
