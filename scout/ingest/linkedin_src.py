"""Future hook (spec §8): LinkedIn ingestion. Stub only — do not build."""

from __future__ import annotations

from scout.config import Seeds, Settings
from scout.ingest.base import SourceAdapter
from scout.models import Account, Tweet
from scout.store import Store


class LinkedInSource(SourceAdapter):
    name = "linkedin"

    def __init__(self, settings: Settings, store: Store) -> None:
        self.settings = settings
        self.store = store

    async def fetch_accounts(
        self, seeds: Seeds, *, max_accounts: int
    ) -> list[Account]:
        raise NotImplementedError("LinkedIn ingestion is a future hook")

    async def fetch_tweets(self, account: Account, *, limit: int = 20) -> list[Tweet]:
        raise NotImplementedError("LinkedIn ingestion is a future hook")

    async def fetch_account(self, handle: str) -> Account | None:
        raise NotImplementedError("LinkedIn ingestion is a future hook")
