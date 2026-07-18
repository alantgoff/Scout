"""Tests for the heuristic regexes — these encode the thesis and get iterated."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from scout.config import Thesis
from scout.models import Account, Signal, Tweet
from scout.signals.heuristics import run_heuristics


def make_thesis() -> Thesis:
    return Thesis(
        thesis="Technical founders leaving top AI labs to build AI infra/devtools",
        keywords=["stealth", "building something new", "day 1", "founding engineer"],
        target_bios=["ex-OpenAI", "ex-Anthropic", "ex-DeepMind", "prev @"],
        sectors=["ai infra", "devtools"],
        disqualifiers=["crypto airdrop", "NFT", "follow back", "growth hacker"],
        weights={
            "bio_intent": 25,
            "smart_money_follow": 25,
            "departure_signal": 20,
            "launch_traction": 15,
            "builder_evidence": 15,
        },
    )


def make_account(**overrides: object) -> Account:
    defaults: dict[str, object] = {"id": "1", "handle": "someone", "followers": 1000}
    defaults.update(overrides)
    return Account.model_validate(defaults)


def make_tweet(
    text: str, *, days_ago: float = 1, likes: int = 0, tweet_id: str = "t1"
) -> Tweet:
    return Tweet(
        id=tweet_id,
        account_id="1",
        text=text,
        created_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        likes=likes,
    )


def get_signal(signals: list[Signal], name: str) -> Signal:
    return next(s for s in signals if s.name == name)


THESIS = make_thesis()


# --- bio_intent -------------------------------------------------------------


def test_bio_intent_matches_thesis_keywords() -> None:
    account = make_account(bio="In stealth. Building something new in AI infra.")
    signals, _ = run_heuristics(account, [], THESIS)
    signal = get_signal(signals, "bio_intent")
    assert signal.value == 1.0
    assert "stealth" in signal.detail
    assert "building something new" in signal.detail


def test_bio_intent_empty_bio_no_hit() -> None:
    account = make_account(bio="")
    signals, _ = run_heuristics(account, [], THESIS)
    assert get_signal(signals, "bio_intent").value == 0.0


# --- departure_signal -------------------------------------------------------


@pytest.mark.parametrize("marker", ["ex-OpenAI", "Ex-OpenAI", "EX-OPENAI", "ex-openai"])
def test_departure_signal_matches_case_variants(marker: str) -> None:
    account = make_account(bio=f"{marker}, now building agents")
    signals, _ = run_heuristics(account, [], THESIS)
    signal = get_signal(signals, "departure_signal")
    assert signal.value == 1.0
    assert "ex-OpenAI" in signal.detail


def test_departure_signal_no_hit_for_unrelated_bio() -> None:
    account = make_account(bio="Coffee enthusiast and weekend cyclist")
    signals, _ = run_heuristics(account, [], THESIS)
    assert get_signal(signals, "departure_signal").value == 0.0


# --- disqualifier -----------------------------------------------------------


@pytest.mark.parametrize("bio", ["Minting my NFT drop soon", "nft degen", "Nft artist"])
def test_disqualifier_nft_any_case(bio: str) -> None:
    account = make_account(bio=bio)
    _, disqualified = run_heuristics(account, [], THESIS)
    assert disqualified is True


def test_clean_bio_not_disqualified() -> None:
    account = make_account(bio="ex-OpenAI, building something new")
    _, disqualified = run_heuristics(account, [], THESIS)
    assert disqualified is False


# --- builder_evidence -------------------------------------------------------


def test_builder_evidence_github_link_in_bio() -> None:
    account = make_account(bio="Open source at github.com/someone")
    signals, _ = run_heuristics(account, [], THESIS)
    signal = get_signal(signals, "builder_evidence")
    assert signal.value == 1.0
    assert "github.com" in signal.detail


def test_builder_evidence_twitter_link_alone_no_hit() -> None:
    account = make_account(
        bio="Find me at twitter.com/someone",
        website="https://twitter.com/someone",
    )
    signals, _ = run_heuristics(account, [], THESIS)
    assert get_signal(signals, "builder_evidence").value == 0.0


def test_builder_evidence_unexpanded_tco_link_no_hit() -> None:
    account = make_account(website="https://t.co/AbC123xyz")
    signals, _ = run_heuristics(account, [], THESIS)
    assert get_signal(signals, "builder_evidence").value == 0.0


# --- launch_traction --------------------------------------------------------


def test_launch_traction_recent_launch_tweet_with_engagement() -> None:
    account = make_account(followers=1000)
    # ratio 80/1000 = 0.08 > the 5% floor
    tweet = make_tweet("Launching our eval platform today!", days_ago=1, likes=80)
    signals, _ = run_heuristics(account, [tweet], THESIS)
    signal = get_signal(signals, "launch_traction")
    assert signal.value > 0.0
    assert tweet.id in signal.detail


def test_launch_traction_old_tweet_scores_zero() -> None:
    account = make_account(followers=1000)
    tweet = make_tweet("Launching our eval platform today!", days_ago=60, likes=80)
    signals, _ = run_heuristics(account, [tweet], THESIS)
    assert get_signal(signals, "launch_traction").value == 0.0


def test_launch_traction_uses_thesis_launch_phrases() -> None:
    thesis = make_thesis()
    thesis.launch_phrases = ["going live"]
    account = make_account(followers=1000)
    custom = make_tweet("We're going live today!", days_ago=1, likes=80)
    default_phrase = make_tweet(
        "Launching our eval platform today!", days_ago=1, likes=80, tweet_id="t2"
    )

    signals, _ = run_heuristics(account, [custom], thesis)
    assert get_signal(signals, "launch_traction").value > 0.0

    # The stock phrase no longer fires once the thesis overrides the list.
    signals, _ = run_heuristics(account, [default_phrase], thesis)
    assert get_signal(signals, "launch_traction").value == 0.0


def test_launch_traction_default_phrases_boundary_guarded() -> None:
    # Default Thesis carries the stock phrase list; "day 10" must not match "day 1".
    thesis = make_thesis()
    account = make_account(followers=1000)
    tweet = make_tweet("day 10 of my job hunt", days_ago=1, likes=80)
    signals, _ = run_heuristics(account, [tweet], thesis)
    assert get_signal(signals, "launch_traction").value == 0.0


def test_launch_traction_empty_phrases_scores_zero() -> None:
    thesis = make_thesis()
    thesis.launch_phrases = []
    account = make_account(followers=1000)
    tweet = make_tweet("Launching our eval platform today!", days_ago=1, likes=80)
    signals, _ = run_heuristics(account, [tweet], thesis)
    assert get_signal(signals, "launch_traction").value == 0.0


# --- run_heuristics ---------------------------------------------------------


def test_run_heuristics_empty_tweets_returns_all_signals() -> None:
    account = make_account(bio="")
    signals, disqualified = run_heuristics(account, [], THESIS)
    assert {s.name for s in signals} == {
        "bio_intent",
        "departure_signal",
        "launch_traction",
        "smart_money_follow",
        "smart_money_convergence",
        "bio_change",
        "builder_evidence",
        "github_evidence",
        "source_corroboration",
    }
    assert disqualified is False


def test_source_corroboration_counts_distinct_sources() -> None:
    single = make_account(bio="", source="search")
    multi = make_account(bio="", source="search", sources=["search", "github", "hn"])
    single_sig = get_signal(run_heuristics(single, [], THESIS)[0], "source_corroboration")
    multi_sig = get_signal(run_heuristics(multi, [], THESIS)[0], "source_corroboration")
    assert single_sig.value == 0.0  # one source is the baseline
    assert multi_sig.value == 1.0  # three distinct sources = full credit
    assert "github" in multi_sig.detail


# --- new v2 signals -----------------------------------------------------------


def test_smart_money_convergence_two_recent_watchers_full_credit() -> None:
    account = make_account(bio="")
    account.recent_followed_by = ["eladgil", "saranormous"]
    signals, _ = run_heuristics(account, [], THESIS)
    assert get_signal(signals, "smart_money_convergence").value == 1.0


def test_smart_money_convergence_one_recent_watcher_half_credit() -> None:
    account = make_account(bio="")
    account.recent_followed_by = ["eladgil"]
    signals, _ = run_heuristics(account, [], THESIS)
    assert get_signal(signals, "smart_money_convergence").value == 0.5


def test_smart_money_convergence_old_follows_dont_count() -> None:
    account = make_account(bio="")
    account.followed_by = ["eladgil", "saranormous"]  # total, any age
    account.recent_followed_by = []  # nothing recent
    signals, _ = run_heuristics(account, [], THESIS)
    assert get_signal(signals, "smart_money_convergence").value == 0.0
    assert get_signal(signals, "smart_money_follow").value > 0.0


def test_bio_change_signal_reads_enrichment_flag() -> None:
    account = make_account(bio="stealth")
    account.bio_changed = True
    signals, _ = run_heuristics(account, [], THESIS)
    assert get_signal(signals, "bio_change").value == 1.0
    account.bio_changed = False
    signals, _ = run_heuristics(account, [], THESIS)
    assert get_signal(signals, "bio_change").value == 0.0


def test_github_evidence_from_repo_field() -> None:
    account = make_account(bio="")
    account.github_repo = "https://github.com/foo/agent-kit"
    signals, _ = run_heuristics(account, [], THESIS)
    signal = get_signal(signals, "github_evidence")
    assert signal.value == 1.0
    assert "agent-kit" in signal.detail


def test_intent_appeared_detects_new_stealth_language() -> None:
    from scout.signals.heuristics import intent_appeared

    assert intent_appeared("ml engineer at BigCo", "stealth — building", THESIS)
    # already present before → not a change
    assert not intent_appeared("stealth founder", "stealth builder", THESIS)
    # first sighting → no baseline, no change
    assert not intent_appeared(None, "stealth", THESIS)
    # keyword disappeared → not a change either
    assert not intent_appeared("stealth", "ml engineer", THESIS)
