"""Which thesis is this caller working against?

Before multiplayer there was one answer: whatever `thesis.yaml` held. That
breaks the moment two people share an installation — one partner exploring
a new space would silently re-aim the other partner's workspace, and worse,
the scheduled run, which has no human to notice.

So there are now three pointers, in precedence order:

1. An explicit `thesis_id` (a CLI flag, or a job payload) — always wins.
2. The member's own pointer (`users.active_thesis_id`) — what the UI sets
   when someone switches, visible only to them.
3. The workspace default (`theses.is_active`) — what unattended work uses,
   because a scheduled run belongs to the firm, not to whoever last clicked
   Switch.

`thesis.yaml` remains the fallback beneath all three, so a single-user
install with no database rows behaves exactly as it always did.

The database is the source of truth for configuration; the YAML files are
kept in sync as a readable export and an editing surface.
"""

from __future__ import annotations

from pathlib import Path

from scout.config import (
    Thesis,
    ensure_thesis_id,
    load_thesis,
    save_thesis,
    thesis_path,
)
from scout.store import Store

DEFAULT_PATH = Path("thesis.yaml")


def resolve_id(
    store: Store,
    *,
    actor: str | None = None,
    thesis_id: str | None = None,
) -> str | None:
    """Apply the precedence order above. None means "fall back to the file"."""
    if thesis_id:
        return thesis_id.strip() or None
    if actor:
        mine = store.user_thesis_id(actor)
        if mine:
            return mine
    return store.active_thesis_id()


def resolve(
    store: Store,
    *,
    actor: str | None = None,
    thesis_id: str | None = None,
    path: Path = DEFAULT_PATH,
) -> Thesis:
    """The Thesis this caller should work against.

    Resolution never fails: an id that no longer has a stored config (a
    thesis backfilled from run history, say) falls through to the file
    rather than raising, because refusing to load anything would take the
    whole workspace down over a dangling pointer.
    """
    wanted = resolve_id(store, actor=actor, thesis_id=thesis_id)
    if wanted:
        config = store.get_thesis_config(wanted)
        if config:
            try:
                return Thesis.model_validate(config)
            except Exception:  # noqa: BLE001 — a corrupt row must not brick the app
                pass
        library = thesis_path(wanted, path)
        if library.exists():
            return load_thesis(library)
    return load_thesis(path)


def persist(
    store: Store,
    thesis: Thesis,
    *,
    path: Path = DEFAULT_PATH,
    write_active_file: bool = True,
) -> str:
    """Save a thesis to the database AND the YAML library. Returns its id.

    Both, deliberately. The database is what the firm shares and what gets
    backed up; the files are what a human reads, diffs, and edits in an
    editor. `write_active_file=False` skips touching thesis.yaml, which is
    what a per-user switch wants — that member's choice must not re-aim
    everyone else's runs.
    """
    thesis_id = thesis.id or ensure_thesis_id(thesis)
    thesis.id = thesis_id
    store.save_thesis_config(thesis_id, thesis.model_dump(mode="json"))
    if write_active_file:
        save_thesis(thesis, path)
    else:
        library = thesis_path(thesis_id, path)
        library.parent.mkdir(parents=True, exist_ok=True)
        save_thesis(thesis, library)
    return thesis_id


def switch_for_user(store: Store, actor: str, thesis_id: str) -> None:
    """Point one member at a thesis. Their own view only."""
    store.set_user_thesis(actor, thesis_id)


def set_workspace_default(store: Store, thesis_id: str) -> None:
    """Set what unattended work (scheduled runs, digests) sources against."""
    store.set_active_thesis(thesis_id)


def sync_files_to_db(store: Store, path: Path = DEFAULT_PATH) -> int:
    """Import YAML theses that have no stored configuration yet.

    The upgrade path for an install whose theses only ever lived on disk.
    Idempotent and non-destructive: a thesis already carrying a config in
    the database is left alone, so this can run on every startup without
    overwriting edits made through the UI.
    """
    imported = 0
    candidates: list[Path] = []
    if path.exists():
        candidates.append(path)
    library_dir = thesis_path("x", path).parent
    if library_dir.exists():
        candidates.extend(sorted(library_dir.glob("*.yaml")))

    seen: set[str] = set()
    for candidate in candidates:
        try:
            thesis = load_thesis(candidate)
        except Exception:  # noqa: BLE001 — one bad file must not stop the rest
            continue
        thesis_id = thesis.id or ensure_thesis_id(thesis)
        if thesis_id in seen:
            continue
        seen.add(thesis_id)
        if store.get_thesis_config(thesis_id) is not None:
            continue
        store.save_thesis_config(thesis_id, thesis.model_dump(mode="json"))
        # Register it too, so a thesis that exists only as a file still
        # appears in the picker with a name rather than a bare slug.
        store.upsert_thesis(
            thesis_id, name=thesis.name or thesis_id,
            statement=thesis.thesis, version="",
        )
        imported += 1
    return imported
