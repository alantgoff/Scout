# Scout, self-hosted: one image for the workspace and the worker.
# Build and run with `docker compose up -d` — see deploy/docker/README.md.
#
# The image holds code only. Everything the firm owns (database, thesis,
# seeds, logs, exports) lives in the /data volume, so rebuilding or
# upgrading never touches it. Secrets arrive as environment variables from
# scout.env at run time and are never baked into a layer (.dockerignore
# keeps .env, cookies and databases out of the build context).
FROM python:3.12-slim

# git: `scout publish --push` (phone app via GitHub Pages) shells out to it.
RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.8.17
ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /app
# Dependencies first, so a code change doesn't reinstall them.
COPY pyproject.toml uv.lock .python-version ./
# The uv cache is a BuildKit cache mount: fast rebuilds, and not ~500MB
# of downloaded wheels left behind in a layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY . .
# Editable on purpose: the app locates thesis.yaml, seeds.yaml and docs/
# relative to its own source tree, which must therefore stay at /app.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev \
 && python -m scout.container layout \
 && python -m compileall -q scout

RUN useradd --uid 10001 --create-home --home-dir /home/scout --shell /usr/sbin/nologin scout \
 && mkdir -p /data && chown scout:scout /data
USER scout
ENV HOME=/home/scout \
    SCOUT_DATA=/data \
    DB_PATH=/data/scout.db \
    OUT_DIR=/data/out
VOLUME ["/data"]
EXPOSE 8501

ENTRYPOINT ["python", "-m", "scout.container"]
CMD ["ui"]
