"""Server configuration shared by players and environments."""

from poke_env.ps_client import ServerConfiguration

from psbot.config import Settings


def build_server_configuration(settings: Settings | None = None) -> ServerConfiguration:
    """Build a poke-env server configuration from PSBot settings."""

    resolved = settings or Settings()
    return ServerConfiguration(resolved.showdown_ws, resolved.showdown_action_url)
