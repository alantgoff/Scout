"""The self-host container's entrypoint: `python -m scout.container <cmd>`.

The image holds code; the volume (/data) holds everything the firm owns —
the database, the thesis and seeds, logs, exports, the rendered phone app.
Recreating or upgrading the container must never touch any of it, so the
user-owned files the app reads relative to its own root (thesis.yaml,
seeds.yaml, the theses/ library, docs/) are symlinks baked into the image
that point into the volume, and the shipped defaults are copied there once,
on first start, never over an existing file.

Commands:
  ui        the Streamlit workspace on 0.0.0.0:8501
  worker    the job loop (bootstraps the default schedules on a fresh DB)
  layout    build-time only: move defaults aside, plant the symlinks
  <other>   passed to the CLI — `docker compose run --rm worker doctor`

Two guards live here because the container is where they can be enforced:
a workspace reachable from outside the host refuses to start without Google
sign-in (an open Scout is an open door to the firm's deal flow), and the
schedules are bootstrapped only into a database that has never had any, so
a schedule an operator deleted stays deleted across restarts.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import sys
import tempfile
from pathlib import Path

APP = Path(__file__).resolve().parent.parent
DEFAULTS_DIR = "defaults"

# Path the app reads (relative to APP) → where it lives in the volume.
LINKS: dict[str, str] = {
    "thesis.yaml": "config/thesis.yaml",
    "seeds.yaml": "config/seeds.yaml",
    "theses": "config/theses",
    "outcomes.yaml": "config/outcomes.yaml",
    "docs": "docs",
}
# The subset with shipped defaults, copied into the volume on first start.
SEEDED = ("thesis.yaml", "seeds.yaml", "theses")
DATA_DIRS = ("config", "out", "logs", "docs")

BOOTSTRAP_MARKER = ".schedules-bootstrapped"
COOKIE_SECRET_FILE = ".cookie-secret"
LOOPBACK = {"127.0.0.1", "localhost", "::1"}

UI_PORT = "8501"


def data_dir(env: dict[str, str] | None = None) -> Path:
    return Path((env if env is not None else os.environ).get("SCOUT_DATA", "/data"))


# ---------------------------------------------------------------- build time


def layout_image(app: Path = APP, data: Path = Path("/data")) -> list[str]:
    """Move the shipped defaults aside and point the app's paths at the
    volume. Run once, as root, while the image is built. Idempotent."""
    defaults = app / DEFAULTS_DIR
    defaults.mkdir(exist_ok=True)
    planted = []
    for name, target in LINKS.items():
        here = app / name
        if here.is_symlink():
            continue
        if here.exists():
            if name in SEEDED:
                shutil.move(str(here), str(defaults / name))
            elif here.is_dir():
                shutil.rmtree(here)
            else:
                here.unlink()
        here.symlink_to(data / target)
        planted.append(name)
    return planted


# ---------------------------------------------------------------- run time


def seed_data(data: Path, defaults: Path) -> list[str]:
    """Create the volume's layout and copy in any default the firm doesn't
    have yet. Never overwrites: the volume's copy is the firm's.

    Both containers call this at start, possibly at the same moment, so
    each copy lands under a temporary name and is renamed into place."""
    for sub in DATA_DIRS:
        (data / sub).mkdir(parents=True, exist_ok=True)
    seeded = []
    for name in SEEDED:
        dest = data / LINKS[name]
        src = defaults / name
        if dest.exists() or not src.exists():
            continue
        tmp = Path(tempfile.mkdtemp(dir=dest.parent, prefix=f".{dest.name}."))
        try:
            staged = tmp / dest.name
            if src.is_dir():
                shutil.copytree(src, staged)
            else:
                shutil.copy2(src, staged)
            try:
                staged.rename(dest)
                seeded.append(name)
            except OSError:
                pass  # the other container won the race — theirs is identical
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    return seeded


def is_public(env: dict[str, str]) -> bool:
    """Whether the workspace can be reached from beyond this host: served
    under a domain (the tls profile), or its port bound off loopback."""
    if (env.get("SCOUT_DOMAIN") or "").strip():
        return True
    bind = (env.get("SCOUT_BIND") or "127.0.0.1").strip()
    return bind not in LOOPBACK


def cookie_secret(env: dict[str, str], data: Path) -> str:
    """SCOUT_COOKIE_SECRET, else one generated on first start and kept in
    the volume — rotating it on every restart would sign everyone out."""
    given = (env.get("SCOUT_COOKIE_SECRET") or "").strip()
    if given:
        return given
    path = data / COOKIE_SECRET_FILE
    if path.exists():
        return path.read_text().strip()
    value = secrets.token_hex(32)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as fh:
        fh.write(value)
    return value


def render_secrets(env: dict[str, str], secret: str) -> str | None:
    """Streamlit's [auth] block from the environment, or None when Google
    sign-in isn't configured. Values are JSON-quoted, which is valid TOML
    for any string a client id or secret can hold."""
    client_id = (env.get("GOOGLE_CLIENT_ID") or "").strip()
    client_secret = (env.get("GOOGLE_CLIENT_SECRET") or "").strip()
    if not (client_id and client_secret):
        return None
    domain = (env.get("SCOUT_DOMAIN") or "").strip()
    redirect = (env.get("SCOUT_REDIRECT_URI") or "").strip() or (
        f"https://{domain}/oauth2callback" if domain
        else f"http://localhost:{env.get('SCOUT_PORT') or UI_PORT}/oauth2callback"
    )
    fields = {
        "redirect_uri": redirect,
        "cookie_secret": secret,
        "client_id": client_id,
        "client_secret": client_secret,
        "server_metadata_url":
            "https://accounts.google.com/.well-known/openid-configuration",
    }
    return "[auth]\n" + "".join(f"{k} = {json.dumps(v)}\n" for k, v in fields.items())


def mounted_auth(app: Path, home: Path) -> bool:
    """An [auth] block in a secrets.toml the operator mounted themselves."""
    for path in (app / ".streamlit" / "secrets.toml", home / ".streamlit" / "secrets.toml"):
        try:
            if any(line.strip() == "[auth]" for line in path.read_text().splitlines()):
                return True
        except OSError:
            continue
    return False


def auth_problem(env: dict[str, str], has_auth: bool) -> str | None:
    if has_auth or not is_public(env):
        return None
    where = (f"https://{env['SCOUT_DOMAIN'].strip()}" if (env.get("SCOUT_DOMAIN") or "").strip()
             else f"{env.get('SCOUT_BIND')}:{env.get('SCOUT_PORT') or UI_PORT}")
    return (
        f"Refusing to serve Scout at {where} without sign-in — anyone who can "
        "reach it would see the firm's deal flow. Set GOOGLE_CLIENT_ID and "
        "GOOGLE_CLIENT_SECRET in scout.env (deploy/docker/README.md, step 3), "
        "or leave SCOUT_DOMAIN unset and SCOUT_BIND=127.0.0.1 and reach it "
        "over SSH or a VPN."
    )


def should_bootstrap(data: Path, schedule_count: int) -> bool:
    """Default schedules go only into a database that has never had any.
    A migrated DB with its own schedules — or with some deliberately
    deleted — is left exactly as it is."""
    return not (data / BOOTSTRAP_MARKER).exists() and schedule_count == 0


def bootstrap(data: Path, env: dict[str, str]) -> str:
    from scout.config import Settings
    from scout.store import Store
    from scout.worker import bootstrap_schedules

    if (env.get("SCOUT_BOOTSTRAP") or "true").strip().lower() in ("0", "false", "no"):
        return "schedule bootstrap disabled (SCOUT_BOOTSTRAP=false)"
    marker = data / BOOTSTRAP_MARKER
    if marker.exists():
        return ""
    store = Store(Settings().db_path, actor="system:scout")
    count = len(store.schedules())
    message = ""
    if should_bootstrap(data, count):
        created = bootstrap_schedules(store)
        message = f"created {len(created)} default schedule(s) — edit them under Automation"
    marker.write_text("1\n")
    return message


def command_for(argv: list[str], app: Path = APP) -> list[str]:
    cmd, *rest = argv or ["ui"]
    if cmd == "ui":
        return [sys.executable, "-m", "streamlit", "run", str(app / "scout" / "ui.py"),
                "--server.address", "0.0.0.0", "--server.port", UI_PORT,
                "--server.headless", "true", *rest]
    if cmd == "worker":
        return [sys.executable, "-m", "scout.cli", "worker", *rest]
    return [sys.executable, "-m", "scout.cli", cmd, *rest]


def _say(message: str) -> None:
    print(f"scout: {message}", file=sys.stderr, flush=True)


def main(argv: list[str]) -> int:
    if argv[:1] == ["layout"]:
        for name in layout_image():
            _say(f"linked {name} → /data/{LINKS[name]}")
        return 0
    env = dict(os.environ)
    data = data_dir(env)
    if not os.access(data, os.W_OK):
        _say(f"{data} is not writable by uid {os.getuid()} — mount a named volume "
             "there (compose.yaml does), or chown a bind mount to 10001.")
        return 1
    for name in seed_data(data, APP / DEFAULTS_DIR):
        _say(f"first start — seeded {data / LINKS[name]} from the shipped default")
    cmd = (argv or ["ui"])[0]
    if cmd == "ui":
        home = Path(env.get("HOME") or "/home/scout")
        configured = all((env.get(k) or "").strip()
                         for k in ("GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET"))
        rendered = render_secrets(env, cookie_secret(env, data)) if configured else None
        if rendered:
            path = home / ".streamlit" / "secrets.toml"
            path.parent.mkdir(parents=True, exist_ok=True)
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w") as fh:
                fh.write(rendered)
        problem = auth_problem(env, bool(rendered) or mounted_auth(APP, home))
        if problem:
            _say(problem)
            return 1
    elif cmd == "worker":
        message = bootstrap(data, env)
        if message:
            _say(message)
    args = command_for(argv)
    os.execv(args[0], args)
    return 0  # unreachable


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
