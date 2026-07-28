"""The safety net and the trace lifecycle are the point of PSBotPlayer.

Battles are real (offline) poke_env Battle objects: choose_random_move
type-checks its argument, so a plain fake would not exercise the real
fallback path. Players are built with start_listening=False, poke-env's
official no-network construction.
"""

import logging

from poke_env.battle import AbstractBattle, Battle

from psbot.agents.base import Decision, PSBotPlayer
from psbot.constants import BATTLE_FORMAT


class RaisingPlayer(PSBotPlayer):
    """A strategy whose decide() always crashes."""

    def decide(self, battle: AbstractBattle) -> Decision:
        raise KeyError("boom")


class ScriptedPlayer(PSBotPlayer):
    """A strategy that always picks the random-legal order with a fixed reason."""

    def decide(self, battle: AbstractBattle) -> Decision:
        return Decision(self.choose_random_move(battle), "scripted reason")


def make_battle(tag: str = "battle-gen9randombattle-1") -> Battle:
    return Battle(tag, "tester", logging.getLogger("test"), gen=9)


def make_player(cls: type[PSBotPlayer]) -> PSBotPlayer:
    return cls(battle_format=BATTLE_FORMAT, start_listening=False)


def test_decide_crash_falls_back_and_confesses() -> None:
    player = make_player(RaisingPlayer)
    battle = make_battle()

    order = player.choose_move(battle)

    assert "default" in str(order)  # empty battle: random-legal is /choose default
    (entry,) = player.peek_traces(battle.battle_tag)
    assert entry.reason.startswith("FALLBACK random:")
    assert "KeyError" in entry.reason


def test_empty_battle_yields_legal_default_order() -> None:
    player = make_player(ScriptedPlayer)
    battle = make_battle()  # no available moves or switches

    order = player.choose_move(battle)

    assert "default" in str(order)
    (entry,) = player.peek_traces(battle.battle_tag)
    assert entry.reason == "scripted reason"


def test_drain_clears_only_that_battle() -> None:
    player = make_player(ScriptedPlayer)
    first = make_battle("battle-gen9randombattle-1")
    second = make_battle("battle-gen9randombattle-2")
    player.choose_move(first)
    player.choose_move(first)
    player.choose_move(second)

    drained = player.drain_traces(first.battle_tag)

    assert len(drained) == 2
    assert player.drain_traces(first.battle_tag) == ()  # gone after drain
    assert len(player.peek_traces(second.battle_tag)) == 1  # untouched


def test_peek_does_not_clear() -> None:
    player = make_player(ScriptedPlayer)
    battle = make_battle()
    player.choose_move(battle)

    assert len(player.peek_traces(battle.battle_tag)) == 1
    assert len(player.peek_traces(battle.battle_tag)) == 1
