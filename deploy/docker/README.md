# Self-hosting Scout with Docker Compose

For a firm that won't put deal flow on someone else's server. Everything runs
on one machine you control: the workspace, the job worker, and optionally
TLS (Caddy) and continuous backup (Litestream). The only outbound traffic is
what a sourcing run makes. See "What leaves the box" below.

```
compose.yaml              ui + worker (+ caddy, litestream behind profiles)
Dockerfile                one image for the workspace and the worker
deploy/docker/
  scout.env.example       → copy to ./scout.env (gitignored, never in the image)
  Caddyfile               TLS for SCOUT_DOMAIN       (profile: tls)
  litestream.yml          replication to S3/R2/B2    (profile: backup)
```

**The volume is the firm's; the image is just code.** Everything Scout writes
lives in the `scout-data` volume at `/data`: the database, thesis and seeds,
the thesis library, logs, CSV exports, the rendered phone app, and the sign-in
cookie secret. Rebuilding, upgrading or deleting the containers never touches
it. On first start the shipped `thesis.yaml`, `seeds.yaml` and `theses/` are
copied into `/data/config`. After that they are yours, and an upgrade never
overwrites them.

## 1. Start it

```bash
git clone https://github.com/alantgoff/Scout.git && cd Scout
cp deploy/docker/scout.env.example scout.env      # add ANTHROPIC_API_KEY, SEC_USER_AGENT
docker compose up -d
docker compose exec worker scout doctor           # what's ready, what to fix
```

Open <http://localhost:8501>. On first start the worker creates the daily
schedules: run 06:00 → resolve 06:30 → refresh 06:45 → digest 07:30 →
phone-app publish 07:45, all UTC. Edit them under **Automation**. This
happens only on a database that has never had schedules, so one you delete
stays deleted across restarts. Set `SCOUT_BOOTSTRAP=false` to skip it.

Run any CLI command inside the running worker:

```bash
docker compose exec worker scout run --strategy github,hn
docker compose exec worker scout jobs
docker compose run --rm worker yield          # or as a one-off container
```

## 2. Reaching it

The workspace binds to **127.0.0.1 only**. Without sign-in, anyone who can
reach the port sees the firm's deal flow, so there are three ways in:

- **SSH tunnel or VPN (simplest):** `ssh -L 8501:localhost:8501 scout-box`,
  then open localhost:8501.
- **Your own domain with HTTPS:** step 3, then step 4.
- **A LAN or VPN address:** set `SCOUT_BIND=10.0.0.5` (in the shell or a
  compose `.env`) and configure sign-in (step 3).

The container enforces this. If `SCOUT_DOMAIN` is set, or `SCOUT_BIND` is
off loopback, the workspace **refuses to start** until Google sign-in is
configured, and its log says why.

## 3. Google sign-in

In Google Cloud Console, create an OAuth client of type *Web application*
with the redirect URI `https://<SCOUT_DOMAIN>/oauth2callback`. Put the client
ID and secret in `scout.env`:

```
SCOUT_DOMAIN=scout.yourfirm.com
GOOGLE_CLIENT_ID=….apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=…
SCOUT_ADMIN_EMAILS=you@yourfirm.com
```

The container writes Streamlit's `[auth]` block from these at start. The
cookie secret is generated once and kept in the volume, so restarts don't
sign everyone out. Set `SCOUT_COOKIE_SECRET` to manage it yourself. Then, as
admin, set the allowed email domain under **Settings → Workspace**; until you
do, anyone who completes Google sign-in is admitted.

You can also mount your own `secrets.toml` at `/app/.streamlit/secrets.toml`.

## 4. HTTPS (profile `tls`)

Point the domain's DNS at the host and open ports 80 and 443. Then:

```bash
docker compose --profile tls up -d
```

Caddy obtains and renews the certificate for `SCOUT_DOMAIN` itself and
proxies the workspace, websocket included.

## 5. Backups (profile `backup`)

Fill in the `LITESTREAM_*` values in `scout.env` (any S3-compatible bucket:
AWS S3, Cloudflare R2, Backblaze B2, MinIO), then:

```bash
docker compose --profile backup up -d
```

Litestream replicates every change within about 10 seconds and keeps 30
days. **Rehearse the restore before you need it:**

```bash
docker compose run --rm litestream \
  restore -o /data/restored.db /data/scout.db
```

To roll back for real, stop the stack, move `restored.db` over `scout.db` in
the volume, and start it again.

## 6. Upgrading

```bash
git pull && docker compose up -d --build
```

Schema changes are additive and apply on first start.

## Bringing data in

- **An existing database** (from a laptop or the VM deploy):
  `docker compose cp ~/.scout/scout.db worker:/data/scout.db`, then
  `docker compose restart`.
- **X cookies** (optional; without them runs use the free sources only):
  `docker compose cp x_cookies.json worker:/data/x_cookies.json`, then set
  `TW_COOKIES=/data/x_cookies.json` in `scout.env` and run
  `docker compose up -d --force-recreate`.
- **Bind mount instead of the named volume:** the containers run as uid
  10001, so `chown -R 10001 /srv/scout-data` first.

## What leaves the box

Only calls a run makes on purpose:

- **Anthropic:** classification, research and memos, bounded by
  `DAILY_SPEND_CAP_USD`.
- **The discovery sources:** GitHub, HN, arXiv, SEC EDGAR, the YC
  directory, RSS feeds, X if configured, and the company websites being
  classified.
- **Slack:** only if you set a webhook.

Nothing else phones home. Streamlit's usage statistics are off in
`.streamlit/config.toml`.

**The phone app.** `scout publish` renders it into `/data/docs`.
`DIGEST_REPO` (GitHub Pages) works from the container, because git is in the
image. The Vercel CLI is not, so publish to Vercel from a machine that has it.
A self-hosting firm may prefer to publish nothing at all, and the daily
publish job then only renders.
