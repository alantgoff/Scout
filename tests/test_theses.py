"""Per-member thesis resolution: the fix for a single global thesis.yaml.

The guarantee under test is that one partner exploring a new space cannot
silently re-aim their partner's workspace — or, worse, the unattended
scheduled run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scout import theses as theses_mod
from scout.config import Thesis, ensure_thesis_id, save_thesis
from scout.store import Store


def make_thesis(thesis_id: str, statement: str) -> Thesis:
    return Thesis(id=thesis_id, name=thesis_id.replace("-", " ").title(),
                  thesis=statement)


@pytest.fixture()
def workspace(tmp_path: Path) -> tuple[Store, Path]:
    """A store plus a thesis.yaml, as a fresh install would have."""
    store = Store(tmp_path / "t.db", actor="alan@firm.com")
    store.ensure_user("alan@firm.com", name="Alan")
    store.ensure_user("sara@firm.com", name="Sara")
    path = tmp_path / "thesis.yaml"
    save_thesis(make_thesis("ai-infra", "AI infrastructure"), path)
    return store, path


def test_resolution_falls_back_to_the_file_when_nothing_is_set(workspace) -> None:
    """A single-user install with no pointers behaves exactly as before."""
    store, path = workspace
    resolved = theses_mod.resolve(store, actor="alan@firm.com", path=path)
    assert resolved.thesis == "AI infrastructure"


def test_each_partner_keeps_their_own_thesis(workspace) -> None:
    """The whole point: switching is per member."""
    store, path = workspace
    theses_mod.persist(store, make_thesis("ai-infra", "AI infrastructure"),
                       path=path, write_active_file=False)
    theses_mod.persist(store, make_thesis("climate", "Climate hardware"),
                       path=path, write_active_file=False)

    theses_mod.switch_for_user(store, "sara@firm.com", "climate")
    assert theses_mod.resolve(store, actor="sara@firm.com", path=path).thesis == \
        "Climate hardware"
    # Alan never switched, so he is untouched.
    assert theses_mod.resolve(store, actor="alan@firm.com", path=path).thesis == \
        "AI infrastructure"


def test_unattended_work_uses_the_firm_default_not_a_partner_s_choice(workspace) -> None:
    """A scheduled run has no user. It must source against what the firm
    chose, not whatever the last person to click Switch is looking at."""
    store, path = workspace
    theses_mod.persist(store, make_thesis("ai-infra", "AI infrastructure"),
                       path=path, write_active_file=False)
    theses_mod.persist(store, make_thesis("climate", "Climate hardware"),
                       path=path, write_active_file=False)
    theses_mod.set_workspace_default(store, "ai-infra")
    theses_mod.switch_for_user(store, "sara@firm.com", "climate")

    # No actor = unattended.
    assert theses_mod.resolve(store, path=path).thesis == "AI infrastructure"
    # Sara still sees hers.
    assert theses_mod.resolve(store, actor="sara@firm.com", path=path).thesis == \
        "Climate hardware"


def test_explicit_id_beats_every_pointer(workspace) -> None:
    store, path = workspace
    theses_mod.persist(store, make_thesis("climate", "Climate hardware"),
                       path=path, write_active_file=False)
    theses_mod.set_workspace_default(store, "ai-infra")
    theses_mod.switch_for_user(store, "alan@firm.com", "ai-infra")
    resolved = theses_mod.resolve(store, actor="alan@firm.com",
                                  thesis_id="climate", path=path)
    assert resolved.thesis == "Climate hardware"


def test_a_dangling_pointer_degrades_to_the_file(workspace) -> None:
    """A deleted thesis must not take the workspace down with it."""
    store, path = workspace
    store.set_user_thesis("alan@firm.com", "thesis-that-no-longer-exists")
    resolved = theses_mod.resolve(store, actor="alan@firm.com", path=path)
    assert resolved.thesis == "AI infrastructure"


def test_a_corrupt_stored_config_degrades_to_the_file(workspace) -> None:
    store, path = workspace
    store.db["theses"].upsert(
        {"id": "broken", "config_json": '{"weights": "not a dict at all"}'},
        pk="id", alter=True)
    store.set_user_thesis("alan@firm.com", "broken")
    assert theses_mod.resolve(store, actor="alan@firm.com", path=path).thesis == \
        "AI infrastructure"


def test_persist_writes_both_the_database_and_the_yaml(workspace) -> None:
    """The database is what the firm shares; the file is what a human reads."""
    store, path = workspace
    thesis = make_thesis("climate", "Climate hardware")
    thesis.keywords = ["carbon", "grid"]
    theses_mod.persist(store, thesis, path=path)

    stored = store.get_thesis_config("climate")
    assert stored is not None and stored["keywords"] == ["carbon", "grid"]
    from scout.config import thesis_path as library_path

    assert library_path("climate", path).exists()
    assert path.read_text().find("Climate hardware") > 0  # became the active file


def test_persist_without_the_active_file_leaves_thesis_yaml_alone(workspace) -> None:
    """A member editing their own thesis must not re-aim the workspace."""
    store, path = workspace
    before = path.read_text()
    theses_mod.persist(store, make_thesis("climate", "Climate hardware"),
                       path=path, write_active_file=False)
    assert path.read_text() == before
    assert store.get_thesis_config("climate") is not None


def test_sync_adopts_yaml_theses_and_is_idempotent(tmp_path: Path) -> None:
    """The upgrade path for an install whose theses only lived on disk."""
    store = Store(tmp_path / "t.db", actor="alan@firm.com")
    path = tmp_path / "thesis.yaml"
    save_thesis(make_thesis("ai-infra", "AI infrastructure"), path)
    from scout.config import thesis_path as library_path

    library = library_path("climate", path)
    library.parent.mkdir(parents=True, exist_ok=True)
    save_thesis(make_thesis("climate", "Climate hardware"), library)

    assert theses_mod.sync_files_to_db(store, path) == 2
    assert store.get_thesis_config("ai-infra")["thesis"] == "AI infrastructure"
    assert store.get_thesis_config("climate")["thesis"] == "Climate hardware"
    # Registered too, so both appear in the picker with names.
    assert {t["id"] for t in store.list_theses()} >= {"ai-infra", "climate"}
    # Re-running imports nothing and — crucially — does not overwrite an
    # edit made through the UI since.
    store.save_thesis_config("climate", {**store.get_thesis_config("climate"),
                                         "thesis": "Climate hardware, refined"})
    assert theses_mod.sync_files_to_db(store, path) == 0
    assert store.get_thesis_config("climate")["thesis"] == "Climate hardware, refined"


def test_user_pointer_survives_a_reload(workspace) -> None:
    store, path = workspace
    theses_mod.persist(store, make_thesis("climate", "Climate hardware"),
                       path=path, write_active_file=False)
    theses_mod.switch_for_user(store, "sara@firm.com", "climate")
    reopened = Store(store.db_path)
    assert reopened.user_thesis_id("sara@firm.com") == "climate"
    assert theses_mod.resolve(reopened, actor="sara@firm.com",
                              path=path).thesis == "Climate hardware"


def test_config_records_who_changed_it(workspace) -> None:
    store, path = workspace
    theses_mod.persist(store, make_thesis("climate", "Climate hardware"),
                       path=path, write_active_file=False)
    row = store.get_thesis(  "climate")
    assert row["config_updated_by"] == "alan@firm.com"
    assert row["config_updated_at"]


# --- seeds follow the thesis ----------------------------------------------------


def test_switching_thesis_switches_the_sourcing_config(workspace) -> None:
    """The gap this closes: seeds were global while theses were per-member,
    so switching thesis kept the previous one's query bank, GitHub topics
    and arXiv categories. A deep-systems thesis sweeping cs.DC/cs.AR would
    silently inherit an ML thesis's cs.LG sweep and miss the population it
    was written for — sourcing for the wrong thesis, with nothing to see."""
    from scout.config import Seeds

    store, path = workspace
    theses_mod.persist(
        store, make_thesis("ai-infra", "AI infrastructure"),
        seeds=Seeds(arxiv_categories=["cs.LG"], github_topics=["pytorch"]),
        path=path, write_active_file=False)
    theses_mod.persist(
        store, make_thesis("systems", "Memory fabrics"),
        seeds=Seeds(arxiv_categories=["cs.DC", "cs.AR"], github_topics=["cuda"]),
        path=path, write_active_file=False)

    theses_mod.switch_for_user(store, "sara@firm.com", "systems")
    theses_mod.set_workspace_default(store, "ai-infra")

    # Sara's sourcing follows her thesis…
    sara = theses_mod.resolve_seeds(store, actor="sara@firm.com")
    assert sara.arxiv_categories == ["cs.DC", "cs.AR"]
    assert sara.github_topics == ["cuda"]
    # …and unattended work still uses the firm's.
    firm = theses_mod.resolve_seeds(store)
    assert firm.arxiv_categories == ["cs.LG"]


def test_seeds_fall_back_to_the_file_when_none_are_stored(workspace, tmp_path) -> None:
    """An install that never stored per-thesis seeds behaves as it did."""
    from scout.config import Seeds, save_seeds

    store, path = workspace
    seeds_file = tmp_path / "seeds.yaml"
    save_seeds(Seeds(github_topics=["from-the-file"]), seeds_file)
    theses_mod.persist(store, make_thesis("ai-infra", "AI infra"),
                       path=path, write_active_file=False)  # no seeds passed
    resolved = theses_mod.resolve_seeds(store, thesis_id="ai-infra",
                                        seeds_path=seeds_file)
    assert resolved.github_topics == ["from-the-file"]


def test_corrupt_stored_seeds_degrade_to_the_file(workspace, tmp_path) -> None:
    from scout.config import Seeds, save_seeds

    store, path = workspace
    seeds_file = tmp_path / "seeds.yaml"
    save_seeds(Seeds(github_topics=["fallback"]), seeds_file)
    store.db["theses"].upsert(
        {"id": "broken", "seeds_json": '{"github_topics": "not a list"}'},
        pk="id", alter=True)
    resolved = theses_mod.resolve_seeds(store, thesis_id="broken",
                                        seeds_path=seeds_file)
    assert resolved.github_topics == ["fallback"]
