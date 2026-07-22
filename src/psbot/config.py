"""Environment-backed project settings."""

from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment variables or a local `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PSBOT_",
        extra="ignore",
    )

    showdown_ws: str = "ws://localhost:8000/showdown/websocket"
    showdown_action_url: str = "https://play.pokemonshowdown.com/action.php?"
    username: str | None = None
    password: SecretStr | None = None
    wandb_project: str = "psbot"
    hf_repository: str | None = None
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts")
    logs_dir: Path = Path("logs")

    @property
    def showdown_health_url(self) -> str:
        """Return the HTTP origin corresponding to the configured WebSocket server."""

        parsed = urlsplit(self.showdown_ws)
        scheme = "https" if parsed.scheme == "wss" else "http"
        return urlunsplit((scheme, parsed.netloc, "/", "", ""))
