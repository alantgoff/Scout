"""Synthetic sample accounts for `scout demo` — a $0, offline test of the full
scoring/classification/export pipeline with no live X calls.

Every handle here is obviously fake (real people are never fabricated). The set
is hand-built to exercise each signal and both outcomes (strong leads, weak
non-leads, a disqualified account that gets dropped) — plus a founder+company
account pair (ada_infra + evalhq) that the Startups track folds into one entry.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from scout.models import Account, Tweet


def sample_data(now: datetime) -> tuple[list[Account], list[Tweet]]:
    """Return (accounts, tweets) with tweet timestamps recent relative to `now`."""
    recent = now - timedelta(days=3)  # inside the 30-day launch_traction window
    old = now - timedelta(days=120)  # outside the window

    accounts = [
        Account(
            id="demo-1", handle="ada_infra", name="Ada Lin",
            bio="ex-OpenAI. building something new in AI infra — EvalHQ. stealth over. github.com/adalin",
            website="https://github.com/adalin", followers=4200, following=310,
            source="demo", sources=["search", "github"],  # multi-source corroboration
            followed_by=["swyx", "paulg"],
        ),
        Account(  # the company account of Ada's startup — folds with her in the Startups track
            id="demo-9", handle="evalhq", name="EvalHQ",
            bio="EvalHQ — evals for AI agents in production. Day 1. evalhq.ai",
            website="https://evalhq.ai", followers=850, following=40,
            source="demo", followed_by=["swyx"],
        ),
        Account(
            id="demo-2", handle="eval_erin", name="Erin Cho",
            bio="ex-Meta AI. evals researcher turned founder. day 1. erincho.com",
            website="https://erincho.com", followers=3100, following=280,
            source="demo", followed_by=["paulg"],
        ),
        Account(
            id="demo-3", handle="marco_devtools", name="Marco Reyes",
            bio="prev @ Google Brain. founding engineer working on developer tools.",
            website="https://marcoreyes.dev", followers=1800, following=190,
            source="demo", followed_by=["swyx"],
        ),
        Account(
            id="demo-4", handle="jenny_agents", name="Jenny Park",
            bio="building something new — agents that actually work. github.com/jpark",
            website="https://github.com/jpark", followers=6000, following=520,
            source="demo", followed_by=[],
        ),
        Account(
            id="demo-5", handle="stealth_sam", name="Sam Okafor",
            bio="stealth. ex-Anthropic.",
            website=None, followers=900, following=140,
            source="demo", followed_by=[],
        ),
        Account(
            id="demo-6", handle="lena_launch", name="Lena Vasquez",
            bio="indie hacker, building in public.",
            website="https://lena.build", followers=2200, following=400,
            source="demo", followed_by=[],
        ),
        Account(
            id="demo-7", handle="quiet_quinn", name="Quinn Adler",
            bio="software engineer at BigCo. opinions my own.",
            website=None, followers=500, following=310,
            source="demo", followed_by=[],
        ),
        Account(
            id="demo-8", handle="nft_ned", name="Ned Crypto",
            bio="web3 builder. NFT drops weekly. crypto airdrop soon — follow back!",
            website="https://linktr.ee/nftned", followers=15000, following=14000,
            source="demo", followed_by=[],
        ),
    ]

    tweets = [
        Tweet(id="t1", account_id="demo-1", created_at=recent,
              text="Introducing EvalHQ — our eval platform. Day 1 and the waitlist is open.",
              likes=430, retweets=120, replies=60),
        Tweet(id="t9", account_id="demo-9", created_at=recent,
              text="EvalHQ is live — eval infra for agent teams. We just launched the waitlist.",
              likes=210, retweets=60, replies=25),
        Tweet(id="t2", account_id="demo-2", created_at=recent,
              text="We're launching Tracewell, our evals startup — and hiring a founding engineer.",
              likes=380, retweets=95, replies=40),
        Tweet(id="t3", account_id="demo-3", created_at=recent,
              text="We built KiteCI — a faster CI runner. Shipped the beta today.",
              likes=140, retweets=30, replies=18),
        Tweet(id="t4", account_id="demo-4", created_at=recent,
              text="Launching AgentForge today — try the beta!",
              likes=520, retweets=140, replies=70),
        Tweet(id="t5", account_id="demo-5", created_at=old,
              text="Reading about distributed systems this weekend.",
              likes=12, retweets=1, replies=2),
        Tweet(id="t6", account_id="demo-6", created_at=recent,
              text="Shipped v1! The waitlist got way more signups than expected.",
              likes=260, retweets=55, replies=30),
        Tweet(id="t7", account_id="demo-7", created_at=old,
              text="Coffee is the real backend.",
              likes=8, retweets=0, replies=1),
        Tweet(id="t8", account_id="demo-8", created_at=recent,
              text="New NFT drop live! Airdrop for holders. Follow back!",
              likes=90, retweets=200, replies=15),
    ]
    return accounts, tweets
