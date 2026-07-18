"""Ingestion contracts: SourceAdapter (full X sources) and DiscoverySource
(supplementary free sources that surface accounts but can't fetch tweets)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from scout.config import Seeds, Thesis
from scout.models import Account, Tweet, UnlinkedLead


class SourceAdapter(ABC):
    """One data source (twscrape, X API, ...).

    Implementations must:
    - dedupe accounts by handle, merging `followed_by` lists when the same
      account surfaces via multiple seed strategies
    - upsert everything they fetch into the Store (caller passes it in)
    - wrap network calls with tenacity retry (3 attempts, jittered backoff)
    """

    name: str = "base"
    # Whether fetch_tweets may be called concurrently. Paid adapters must stay
    # False: the budget guard pre-checks cumulative spend before each call, and
    # concurrent calls could race past the cap.
    parallel_safe: bool = False

    @abstractmethod
    async def fetch_accounts(self, seeds: Seeds, *, max_accounts: int) -> list[Account]:
        """Run all seed strategies this source supports; return deduped accounts."""

    @abstractmethod
    async def fetch_tweets(self, account: Account, *, limit: int = 20) -> list[Tweet]:
        """Recent tweets for one account (cache-first where fetching costs money)."""

    @abstractmethod
    async def fetch_account(self, handle: str) -> Account | None:
        """Look up a single account by handle (used by `scout inspect`)."""


class DiscoverySource(ABC):
    """A supplementary discovery source (GitHub, HN, ...): finds candidate
    founders on non-X platforms. When a source can bridge to an X handle it
    returns an Account (tweets come from the X adapter later); when it can't,
    it returns an UnlinkedLead for manual lookup.
    """

    name: str = "discovery"

    @abstractmethod
    async def discover(
        self, seeds: Seeds, thesis: Thesis
    ) -> tuple[list[Account], list[UnlinkedLead]]:
        """Return (bridged accounts, unlinked leads). Must be free to call."""
