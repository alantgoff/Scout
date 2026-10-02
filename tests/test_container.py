"""The self-host bundle: the container entrypoint's logic (pure, on temp
dirs) and static checks over Dockerfile / compose.yaml / the env examples,
so the bundle can't drift from the code it packages. The image itself was
built and exercised by hand (deploy/docker/README.md); these pin the
decisions that make it safe."""

from __future__ import annotations

import os
import re
import tomllib
from pathlib import Path

import pytest
import yaml

from scout import container
from scout.config import THESES_DIR, Settings

ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------- image layout


def _fake_app(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    (app / "theses").mkdir(parents=True)
    (app / "theses" / "climate.yaml").write_text("name: Climate\n")
    (app / "thesis.yaml").write_text("name: Shipped\n")
    (app / "seeds.yaml").write_text("queries: {}\n")
    (app / "docs").mkdir()  # a dev's rendered phone app: not a default
    return app


def test_layout_moves_defaults_aside_and_links_into_the_volume(tmp_path):
    app, data = _fake_app(tmp_path), tmp_path / "data"
    planted = container.layout_image(app, data)
    assert set(planted) == set(container.LINKS)
    for name, target in container.LINKS.items():
        assert (app / name).is_symlink()
        assert os.readlink(app / name) == str(data / target)
    assert (app / "defaults" / "thesis.yaml").read_text() == "name: Shipped\n"
    assert (app / "defaults" / "theses" / "climate.yaml").exists()
    assert not (app / "defaults" / "docs").exists()
    assert container.layout_image(app, data) == []  # idempotent


def test_every_path_the_app_writes_beside_its_source_is_linked():
    # ui.py reads PROJECT_ROOT/thesis.yaml + seeds.yaml; the thesis library
    # sits beside the active file; `publish` renders into docs/. Anything
    # written next to the source and NOT linked would vanish on upgrade.
    for name in ("thesis.yaml", "seeds.yaml", THESES_DIR.name, "docs"):
        assert name in container.LINKS
    ui = (ROOT / "scout" / "ui.py").read_text()
    for path in re.findall(r'PROJECT_ROOT / "([^"]+)"', ui):
        assert path in container.LINKS or path == ".env", path


# ------------------------------------------------------------ first start


def test_seed_copies_missing_defaults_and_never_overwrites(tmp_path):
    app, data = _fake_app(tmp_path), tmp_path / "data"
    container.layout_image(app, data)
    defaults = app / "defaults"
    (data / "config").mkdir(parents=True)
    (data / "config" / "thesis.yaml").write_text("name: The firm's own\n")

    seeded = container.seed_data(data, defaults)

    assert set(seeded) == {"seeds.yaml", "theses"}
    assert (data / "config" / "thesis.yaml").read_text() == "name: The firm's own\n"
    assert (data / "config" / "theses" / "climate.yaml").exists()
    for sub in container.DATA_DIRS:
        assert (data / sub).is_dir()
    # Through the image's symlinks, the app now reads the volume's copies.
    assert (app / "thesis.yaml").read_text() == "name: The firm's own\n"
    assert container.seed_data(data, defaults) == []
    assert not [p for p in (data / "config").iterdir() if p.name.startswith(".")]


# ----------------------------------------------------------------- sign-in


@pytest.mark.parametrize("env, public", [
    ({}, False),
    ({"SCOUT_BIND": "127.0.0.1"}, False),
    ({"SCOUT_BIND": "localhost"}, False),
    ({"SCOUT_BIND": "0.0.0.0"}, True),
    ({"SCOUT_BIND": "10.0.0.5"}, True),
    ({"SCOUT_DOMAIN": "scout.firm.com"}, True),
    ({"SCOUT_DOMAIN": "  "}, False),
])
def test_is_public(env, public):
    assert container.is_public(env) is public


def test_public_without_sign_in_refuses_to_start():
    assert container.auth_problem({}, has_auth=False) is None
    assert container.auth_problem({"SCOUT_BIND": "0.0.0.0"}, has_auth=True) is None
    problem = container.auth_problem({"SCOUT_DOMAIN": "scout.firm.com"}, has_auth=False)
    assert "https://scout.firm.com" in problem and "GOOGLE_CLIENT_ID" in problem
    assert "0.0.0.0:8501" in container.auth_problem({"SCOUT_BIND": "0.0.0.0"}, False)


def test_render_secrets_is_valid_toml_for_awkward_values():
    assert container.render_secrets({}, "s") is None
    assert container.render_secrets({"GOOGLE_CLIENT_ID": "x"}, "s") is None
    env = {"GOOGLE_CLIENT_ID": "id.apps.googleusercontent.com",
           "GOOGLE_CLIENT_SECRET": 'a"b\\c\'d', "SCOUT_DOMAIN": "scout.firm.com"}
    auth = tomllib.loads(container.render_secrets(env, "cookie"))["auth"]
    assert auth["client_secret"] == 'a"b\\c\'d'
    assert auth["redirect_uri"] == "https://scout.firm.com/oauth2callback"
    assert auth["cookie_secret"] == "cookie"
    assert auth["server_metadata_url"].startswith("https://accounts.google.com/")
    env["SCOUT_REDIRECT_URI"] = "https://vpn.firm/oauth2callback"
    assert tomllib.loads(container.render_secrets(env, "c"))["auth"][
        "redirect_uri"] == "https://vpn.firm/oauth2callback"
    local = {"GOOGLE_CLIENT_ID": "i", "GOOGLE_CLIENT_SECRET": "s"}
    assert "http://localhost:8501/oauth2callback" in container.render_secrets(local, "c")


def test_cookie_secret_is_generated_once_and_kept(tmp_path):
    assert container.cookie_secret({"SCOUT_COOKIE_SECRET": "given"}, tmp_path) == "given"
    first = container.cookie_secret({}, tmp_path)
    assert len(first) == 64
    assert container.cookie_secret({}, tmp_path) == first  # stable: no mass sign-out
    assert (tmp_path / container.COOKIE_SECRET_FILE).stat().st_mode & 0o777 == 0o600


def test_mounted_secrets_count_as_sign_in(tmp_path):
    app, home = tmp_path / "app", tmp_path / "home"
    assert not container.mounted_auth(app, home)
    (home / ".streamlit").mkdir(parents=True)
    (home / ".streamlit" / "secrets.toml").write_text('[other]\nx = 1\n')
    assert not container.mounted_auth(app, home)
    (app / ".streamlit").mkdir(parents=True)
    (app / ".streamlit" / "secrets.toml").write_text('[auth]\nclient_id = "x"\n')
    assert container.mounted_auth(app, home)


# --------------------------------------------------------------- schedules


def test_schedules_bootstrap_only_into_a_db_that_never_had_any(tmp_path, monkeypatch):
    from scout.store import Store

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DB_PATH", str(data / "scout.db"))
    assert "created 5" in container.bootstrap(data, {})
    assert (data / container.BOOTSTRAP_MARKER).exists()
    store = Store(data / "scout.db")
    store.delete_schedule(store.schedules()[-1]["id"])  # an operator's choice
    assert container.bootstrap(data, {}) == ""        # restart: stays deleted
    assert len(store.schedules()) == 4


def test_migrated_db_with_its_own_schedules_is_left_alone(tmp_path, monkeypatch):
    from scout import jobs
    from scout.store import Store

    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv("DB_PATH", str(data / "scout.db"))
    store = Store(data / "scout.db")
    store.upsert_schedule("Only a digest", jobs.KIND_DIGEST,
                          jobs.ScheduleSpec(daily_at="08:00", tz="UTC"), {})
    assert container.bootstrap(data, {}) == ""
    assert [s["name"] for s in store.schedules()] == ["Only a digest"]
    assert container.should_bootstrap(tmp_path / "fresh", 0)


def test_bootstrap_can_be_switched_off(tmp_path):
    message = container.bootstrap(tmp_path, {"SCOUT_BOOTSTRAP": "false"})
    assert "disabled" in message
    assert not (tmp_path / container.BOOTSTRAP_MARKER).exists()


def test_command_routing():
    ui = container.command_for(["ui"], Path("/app"))
    assert ui[1:4] == ["-m", "streamlit", "run"] and "/app/scout/ui.py" in ui
    assert ui[ui.index("--server.address") + 1] == "0.0.0.0"
    assert container.command_for([], Path("/app")) == ui
    assert container.command_for(["worker"])[1:] == ["-m", "scout.cli", "worker"]
    assert container.command_for(["doctor", "--offline"])[1:] == [
        "-m", "scout.cli", "doctor", "--offline"]


# ------------------------------------------------------- the bundle itself


def _compose() -> dict:
    return yaml.safe_load((ROOT / "compose.yaml").read_text())


def test_compose_keeps_firm_data_in_the_volume_and_ui_on_loopback():
    services = _compose()["services"]
    for name in ("ui", "worker"):
        svc = services[name]
        assert "scout-data:/data" in svc["volumes"]
        assert svc["environment"]["DB_PATH"] == "/data/scout.db"
        assert svc["environment"]["SCOUT_DATA"] == "/data"
        assert svc["init"] is True
        assert svc["env_file"][0]["path"] == "scout.env"
    assert services["ui"]["ports"] == ["${SCOUT_BIND:-127.0.0.1}:${SCOUT_PORT:-8501}:8501"]
    # The guard must see the same bind the port mapping uses.
    assert services["ui"]["environment"]["SCOUT_BIND"] == "${SCOUT_BIND:-127.0.0.1}"
    assert services["worker"]["stop_grace_period"] == "15m"
    assert services["caddy"]["profiles"] == ["tls"]
    assert services["litestream"]["profiles"] == ["backup"]
    assert "scout-data:/data" in services["litestream"]["volumes"]


def test_dockerfile_matches_compose_and_runs_unprivileged():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "DB_PATH=/data/scout.db" in dockerfile
    assert "python -m scout.container layout" in dockerfile
    assert re.search(r"^USER scout$", dockerfile, re.M)
    assert '"python", "-m", "scout.container"' in dockerfile
    assert "--mount=type=cache,target=/root/.cache/uv" in dockerfile
    assert not re.search(r"COPY\s+\S*\.env", dockerfile)


def test_secrets_never_enter_the_build_context():
    ignored = (ROOT / ".dockerignore").read_text().split()
    for pattern in (".env", "scout.env", "**/*.db", "**/secrets.toml",
                    "**/cookies.json", "**/x_cookies.json", ".git"):
        assert pattern in ignored, pattern
    assert "scout.env" in (ROOT / ".gitignore").read_text().split()


# Env vars read outside Settings (by the container, the store, Litestream,
# or the deploy kit) — anything else in an example must be a Settings field,
# or an operator fills in a variable nothing reads.
NON_SETTINGS_ENV = {
    "SCOUT_ADMIN_EMAILS", "SCOUT_DOMAIN", "SCOUT_COOKIE_SECRET",
    "GOOGLE_CLIENT_ID", "GOOGLE_CLIENT_SECRET",
    "LITESTREAM_BUCKET", "LITESTREAM_REGION", "LITESTREAM_ENDPOINT",
    "LITESTREAM_ACCESS_KEY_ID", "LITESTREAM_SECRET_ACCESS_KEY",
}


@pytest.mark.parametrize("example", [
    ".env.example", "deploy/scout.env.example", "deploy/docker/scout.env.example",
])
def test_env_examples_only_name_variables_something_reads(example):
    fields = {name.upper() for name in Settings.model_fields}
    keys = re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", (ROOT / example).read_text(), re.M)
    assert keys
    unknown = sorted(set(keys) - fields - NON_SETTINGS_ENV)
    assert not unknown, f"{example}: {unknown}"


def test_caddyfiles_use_only_stock_directives():
    # rate_limit is a plugin (caddy-ratelimit); with it, `apt install caddy`
    # and the caddy:2 image both refused to start.
    for path in ("deploy/Caddyfile", "deploy/docker/Caddyfile"):
        text = (ROOT / path).read_text()
        directives = {line.split()[0] for line in text.splitlines()
                      if line.strip() and not line.lstrip().startswith("#")
                      and line.startswith("\t")}
        assert "rate_limit" not in directives, path
