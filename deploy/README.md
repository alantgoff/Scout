# Deploying Scout for a firm (one small VM)

One box runs everything: the Streamlit workspace, the job worker, litestream
backup, and Caddy for TLS. SQLite (WAL mode) is the database — correct at this
team size because every write path in `scout/store.py` runs inside
`BEGIN IMMEDIATE` transactions and all processes share one local disk.

```
/opt/scout                   git checkout, owned by the `scout` user
/var/lib/scout/scout.db      the database (DB_PATH)
/var/lib/scout/logs/         run logs
/etc/scout/scout.env         secrets — root:scout 0640, never UI-editable
```

## 1. Provision

```bash
adduser --system --group --home /var/lib/scout scout
mkdir -p /opt/scout /var/lib/scout/logs /etc/scout
git clone https://github.com/alantgoff/Scout.git /opt/scout
chown -R scout:scout /opt/scout /var/lib/scout
curl -LsSf https://astral.sh/uv/install.sh | sh   # as the scout user
cd /opt/scout && sudo -u scout uv sync
```

## 2. Secrets — `/etc/scout/scout.env`

Copy `deploy/scout.env.example`, fill it in, then `chmod 0640` and
`chown root:scout`. This file is the ONLY place secrets live; the UI shows
presence/absence and can never read them back out or edit them.
Regenerate any key that has ever been pasted into a chat or commit.

Non-secret shared knobs (spend cap, model, run sizes, allowlist) are edited
in the app under **Settings** and stored in the database.

## 3. Google sign-in — `.streamlit/secrets.toml`

Copy `deploy/secrets.toml.example` to `/opt/scout/.streamlit/secrets.toml`
(owned by `scout`, mode 0600). Create an OAuth client (type: Web application)
in Google Cloud Console for your Workspace, with the redirect URI
`https://scout.<your-domain>.com/oauth2callback`.

The FIRST person to sign in becomes admin (or set `SCOUT_ADMIN_EMAILS` in
scout.env). The admin then sets the allowed email domain under
Settings → Workspace — until that's set, anyone who can complete Google
sign-in is admitted, so do it first.

## 4. Services

```bash
cp deploy/scout-ui.service deploy/scout-worker.service /etc/systemd/system/
cp deploy/litestream.yml /etc/litestream.yml     # fill in the S3/B2 bucket
systemctl daemon-reload
systemctl enable --now scout-ui litestream
# scout-worker: enable when the jobs worker ships (Phase 3)
```

Caddy (TLS + reverse proxy — Streamlit's websocket proxies out of the box):

```bash
apt install caddy
cp deploy/Caddyfile /etc/caddy/Caddyfile   # edit the domain
systemctl reload caddy
```

The app binds 127.0.0.1:8501; only Caddy is public.

## 5. Migrating an existing single-user database

```bash
scp ~/.scout/scout.db you@vm:/tmp/scout.db
sudo mv /tmp/scout.db /var/lib/scout/scout.db && sudo chown scout:scout /var/lib/scout/scout.db
```

Schema upgrades are additive and run automatically on first start (WAL,
indexes, actor columns). Keep the pre-copy as the rollback — the old code can
still open the upgraded file.

## 6. Backups

- litestream replicates continuously to the bucket in `/etc/litestream.yml`.
- Belt and suspenders, nightly local snapshot (root crontab):
  `sqlite3 /var/lib/scout/scout.db ".backup /var/lib/scout/backups/scout-$(date +\%a).db"`
- Restore drill: `litestream restore -o /tmp/restored.db <replica-url>` then
  point a local `DB_PATH` at it and check the Startups page renders.
