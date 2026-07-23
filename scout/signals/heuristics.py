"""Deterministic, cheap signal scorers — always run before any LLM call.

Emits ALL signals (value 0.0 when not hit) so `scout inspect` can show a
complete per-signal breakdown. Weights are filled in later by score.py.

Three signals read enrichment fields the pipeline sets from store history
(Account.recent_followed_by / bio_changed / github_repo) rather than raw
adapter data — see cli._enrich_accounts. source_corroboration reads
Account.sources, filled by the pipeline's merge step (cli._merge_accounts).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from urllib.parse import urlparse

from scout.config import Thesis
from scout.models import Account, Signal, Tweet

_GITHUB_LINK_RE = re.compile(r"(?:https?://)?(?:www\.)?github\.com/?\S*", re.IGNORECASE)

# Hosts that don't count as a personal site for builder_evidence.
# "t.co" covers unexpanded X shortlinks — they say nothing about the person.
_NON_PERSONAL_HOSTS = ("twitter.com", "x.com", "linktr.ee", "t.co")

# Traction/convergence constants live in thesis.signal_params (UI-editable);
# the module defaults below are only the pydantic defaults' documentation.


def _term_pattern(term: str) -> str:
    """Word-boundary-aware pattern source: "day 1" must not match "day 10",
    but multi-word phrases still match. \\b only applies where the term edge
    is a word character (so terms like "prev @" stay matchable)."""
    prefix = r"\b" if term[:1].isalnum() else ""
    suffix = r"\b" if term[-1:].isalnum() else ""
    return prefix + re.escape(term) + suffix


def _keyword_pattern(term: str) -> re.Pattern[str]:
    return re.compile(_term_pattern(term), re.IGNORECASE)


@lru_cache(maxsize=16)
def _launch_regex(phrases: tuple[str, ...]) -> re.Pattern[str]:
    """Launch-y language regex, built from thesis.launch_phrases and cached
    per phrase-tuple (the thesis is fixed for the life of a run)."""
    return re.compile(
        "|".join(_term_pattern(p) for p in phrases), re.IGNORECASE
    )


def _bio_intent(account: Account, thesis: Thesis) -> Signal:
    matched = [
        term for term in thesis.keywords if _keyword_pattern(term).search(account.bio)
    ]
    return Signal(
        name="bio_intent",
        value=1.0 if matched else 0.0,
        detail=", ".join(matched),
    )


def _departure_signal(account: Account, thesis: Thesis) -> Signal:
    bio_lower = account.bio.lower()
    matched = [org for org in thesis.target_bios if org.lower() in bio_lower]
    return Signal(
        name="departure_signal",
        value=1.0 if matched else 0.0,
        detail=", ".join(matched),
    )


def _launch_traction(account: Account, tweets: list[Tweet], thesis: Thesis) -> Signal:
    phrases = tuple(thesis.launch_phrases)
    params = thesis.signal_params
    if not phrases:
        return Signal(name="launch_traction", value=0.0)
    launch_re = _launch_regex(phrases)
    cutoff = datetime.now(timezone.utc) - timedelta(days=params.traction_window_days)
    best_ratio = 0.0
    best_tweet_id = ""
    for tweet in tweets:
        if tweet.created_at is None:
            continue
        created = tweet.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)
        if created < cutoff:
            continue
        ratio = tweet.engagement / max(account.followers, 1)
        if (
            ratio > params.traction_floor
            and launch_re.search(tweet.text)
            and ratio > best_ratio
        ):
            best_ratio = ratio
            best_tweet_id = tweet.id
    if not best_tweet_id:
        return Signal(name="launch_traction", value=0.0)
    return Signal(
        name="launch_traction",
        value=min(1.0, best_ratio / params.traction_saturation),
        detail=f"tweet {best_tweet_id} engagement/followers={best_ratio:.3f}",
    )


def _smart_money_follow(account: Account) -> Signal:
    return Signal(
        name="smart_money_follow",
        value=min(len(account.followed_by), 3) / 3,
        detail=", ".join(account.followed_by),
    )


def _smart_money_convergence(account: Account, thesis: Thesis) -> Signal:
    """RECENT follows by watchers — the 'diligence shape'. N+ watchers newly
    following the same account within the window is the strongest X signal
    (N = signal_params.convergence_full_credit); fewer earn partial credit."""
    recent = account.recent_followed_by
    full_at = max(thesis.signal_params.convergence_full_credit, 1)
    value = min(len(recent), full_at) / full_at
    return Signal(
        name="smart_money_convergence",
        value=value,
        detail=", ".join(recent),
    )


def intent_appeared(previous_bio: str | None, new_bio: str, thesis: Thesis) -> bool:
    """True when stealth/intent language (thesis keywords) is present in the
    new bio but was absent from the previous one — the bio-CHANGE moment,
    which is the highest-precision founder signal."""
    if previous_bio is None:
        return False  # first sighting — no change to detect
    def hits(text: str) -> set[str]:
        return {
            term for term in thesis.keywords if _keyword_pattern(term).search(text)
        }
    return bool(hits(new_bio) - hits(previous_bio))


def _bio_change(account: Account) -> Signal:
    return Signal(
        name="bio_change",
        value=1.0 if account.bio_changed else 0.0,
        detail="intent language newly appeared in bio" if account.bio_changed else "",
    )


def _github_evidence(account: Account) -> Signal:
    if account.github_repo:
        return Signal(name="github_evidence", value=1.0, detail=account.github_repo)
    return Signal(name="github_evidence", value=0.0)


def _source_corroboration(account: Account) -> Signal:
    """Independent discovery strategies agreeing on the same account.

    `sources` is filled by the pipeline's merge step (cli._merge_accounts).
    One source is the baseline (value 0); a second independent source earns
    half credit, three or more earn full credit.
    """
    distinct = {s for s in (account.sources or [account.source]) if s}
    extra = max(len(distinct) - 1, 0)
    return Signal(
        name="source_corroboration",
        value=min(extra, 2) / 2,
        detail=", ".join(sorted(distinct)) if extra else "",
    )


def _is_personal_site(website: str) -> bool:
    candidate = website if "://" in website else f"https://{website}"
    parsed = urlparse(candidate)
    if parsed.scheme not in ("http", "https"):
        return False
    host = parsed.netloc.lower().removeprefix("www.")
    if not host:
        return False
    return not any(
        host == blocked or host.endswith(f".{blocked}") for blocked in _NON_PERSONAL_HOSTS
    )


def _builder_evidence(account: Account) -> Signal:
    website = account.website or ""
    if "github.com" in website.lower():
        return Signal(name="builder_evidence", value=1.0, detail=website)
    github_in_bio = _GITHUB_LINK_RE.search(account.bio)
    if github_in_bio:
        return Signal(name="builder_evidence", value=1.0, detail=github_in_bio.group(0))
    if website and _is_personal_site(website):
        return Signal(name="builder_evidence", value=1.0, detail=website)
    return Signal(name="builder_evidence", value=0.0)


def verdict_disqualified(verdict: object | None, thesis: Thesis) -> str | None:
    """The disqualifier hit in what the classifier learned, or None.

    The bio pass in `run_heuristics` runs before classification, so it can only
    see how an account describes itself in a 160-character bio. A company whose
    bio reads "AI inference marketplace" and whose product settles in on-chain
    USDC clears that check and lands in the report — the disqualifying fact
    belongs to the product, and the product is only known afterwards.

    Matched against thesis.product_disqualifiers, NOT thesis.disqualifiers:
    see the note on that field for why the two lists cannot be merged.
    """
    if verdict is None or not thesis.product_disqualifiers:
        return None
    haystack = " ".join(
        text for text in (
            getattr(verdict, "product_summary", "") or "",
            getattr(verdict, "one_line_summary", "") or "",
            getattr(verdict, "sector", "") or "",
            getattr(verdict, "subsector", "") or "",
            getattr(verdict, "business_model", "") or "",
        ) if text
    )
    if not haystack:
        return None
    for term in thesis.product_disqualifiers:
        if _keyword_pattern(term).search(haystack):
            return term
    return None


def run_heuristics(
    account: Account, tweets: list[Tweet], thesis: Thesis
) -> tuple[list[Signal], bool]:
    """Score all deterministic signals; second element is True when the bio
    contains a disqualifier term (caller drops the account entirely)."""
    signals = [
        _bio_intent(account, thesis),
        _departure_signal(account, thesis),
        _launch_traction(account, tweets, thesis),
        _smart_money_follow(account),
        _smart_money_convergence(account, thesis),
        _bio_change(account),
        _builder_evidence(account),
        _github_evidence(account),
        _source_corroboration(account),
    ]
    # Word-boundary matched, like every other bio signal above. A bare
    # substring test drops the account when "VC" appears inside "VC-backed"
    # and when "investor" appears inside "backed by investors" — company bios
    # that describe funding, i.e. precisely the accounts worth keeping.
    disqualified = any(
        _keyword_pattern(term).search(account.bio) for term in thesis.disqualifiers
    )
    return signals, disqualified
