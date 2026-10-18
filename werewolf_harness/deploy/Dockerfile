# Build context: the werewolf_harness folder (see deploy/docker-compose.yml).
#
# Two stages so the runtime image carries no pip cache, no build tools and no
# lockfile tooling -- only a venv and the source. python:*-slim is a multi-arch
# manifest (amd64 + arm64), so the same file builds natively on Apple Silicon;
# every pinned wheel in requirements.lock has an arm64 build.
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS deps
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
RUN python -m venv /opt/venv
COPY requirements.lock /tmp/requirements.lock
# --require-hashes: the image installs exactly the reviewed artefacts, nothing resolved at build time
RUN --mount=type=cache,target=/root/.cache/pip \
    /opt/venv/bin/pip install --require-hashes --no-deps -r /tmp/requirements.lock \
 && /opt/venv/bin/pip uninstall -y -q pip    # 16 MB the running service never uses

FROM python:${PYTHON_VERSION}-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH=/opt/venv/bin:$PATH PYTHONPATH=/app
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin app
COPY --from=deps /opt/venv /opt/venv
# Source stays root-owned: the process only reads it. Its one writable place is /app/data.
COPY . /app/werewolf_harness
RUN mkdir -p /app/data && chown app:app /app/data
WORKDIR /app
USER app
EXPOSE 8402 9100
# stdlib only (no curl in slim). Workers override this with their metrics port in compose.
# --start-interval: probe every second while starting (healthy ~1 s after boot), then every 10 s
HEALTHCHECK --interval=10s --timeout=3s --start-period=30s --start-interval=1s --retries=3 \
  CMD ["python", "-c", "import sys,urllib.request as u; sys.exit(0 if u.urlopen('http://127.0.0.1:8402/healthz', timeout=2).status == 200 else 1)"]
# --timeout-graceful-shutdown: SIGTERM -> lifespan drains (releases leases) before the process exits
CMD ["uvicorn", "werewolf_harness.app.api.main:app", "--host", "0.0.0.0", "--port", "8402", \
     "--proxy-headers", "--forwarded-allow-ips", "*", "--timeout-graceful-shutdown", "20"]
