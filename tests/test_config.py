from psbot.config import Settings


def test_health_url_uses_websocket_origin() -> None:
    settings = Settings(
        showdown_ws="wss://example.test:8443/showdown/websocket",
        _env_file=None,
    )

    assert settings.showdown_health_url == "https://example.test:8443/"
