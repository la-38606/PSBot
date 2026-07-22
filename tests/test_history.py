from psbot.constants import HISTORY_WINDOW
from psbot.features.history import BattleEvent, EventType, HistoryBuffer


def test_history_keeps_the_latest_eight_events() -> None:
    history = HistoryBuffer()
    for turn in range(HISTORY_WINDOW + 2):
        history.append(BattleEvent(turn=turn, actor="p1", event_type=EventType.MOVE))

    snapshot = history.snapshot()
    assert len(snapshot) == HISTORY_WINDOW
    assert snapshot[0].turn == 2
    assert snapshot[-1].turn == HISTORY_WINDOW + 1
