# Run from this folder. The package is imported as `werewolf_harness`, so python runs from the parent.
PY ?= ../.venv/bin/python
ROOT := $(abspath ..)
# .env (if any) feeds both the containers (env_file) and compose's own ${VAR} interpolation
# COMPOSE_FILES: add `-f your-override.yml` for host-specific changes (ports, build network, ...)
COMPOSE_FILES ?= -f deploy/docker-compose.yml
COMPOSE = docker compose $(COMPOSE_FILES) $(if $(wildcard .env),--env-file .env,)
# empty = compose's deploy.replicas (WORKERS from .env, else 2); `make up WORKERS=4` scales
WORKERS ?=
# empty = every service; e.g. SERVICES="migrate api1 api2 worker postgres redis caddy prometheus"
SERVICES ?=
# extra flags for scripts/smoke.py, e.g. SMOKE_ARGS="--grafana ''"
SMOKE_ARGS ?=
# CI builds the image with buildx (layer cache) first and passes BUILD=--no-build
BUILD ?= --build

.PHONY: dev run worker test test-core test-int lint migrate bench chaos faults faults-quick load lock up down down-v ps logs smoke compat compat-db

dev:            ## Postgres + Redis + Jaeger in Docker
	docker compose -f deploy/docker-compose.dev.yml up -d
run:            ## API on :8402 (reads .env)
	cd $(ROOT) && $(PY) -m uvicorn werewolf_harness.app.api.main:app --port 8402 --reload
worker:         ## a model worker (AGENT_EXECUTOR=streams)
	cd $(ROOT) && $(PY) -m werewolf_harness.app.llm.worker
migrate:
	cd $(ROOT) && $(PY) -m alembic -c werewolf_harness/alembic.ini upgrade head
test:           ## everything (integration tests need make dev)
	cd $(ROOT) && $(PY) -m pytest werewolf_harness/tests -q $(PYTEST_ARGS)
test-core:      ## no database needed
	cd $(ROOT) && $(PY) -m pytest werewolf_harness/tests -q --ignore=werewolf_harness/tests/integration $(PYTEST_ARGS)
test-int:       ## only the Postgres/Redis tests (TEST_DATABASE_URL / TEST_REDIS_URL)
	cd $(ROOT) && $(PY) -m pytest werewolf_harness/tests/integration -q $(PYTEST_ARGS)
compat:         ## every stored journal replays under its recorded engine version (CI gate); real logs vs current rules
	cd $(ROOT) && $(PY) -m pytest werewolf_harness/tests/test_engine_compat.py -q
	cd $(ROOT) && $(PY) -m werewolf_harness.scripts.compat_check 'werewolf_harness/tests/corpus/*.json' --fingerprint
compat-db:      ## the same check over every game in Postgres (DATABASE_URL) -- run before deploying a new engine version
	cd $(ROOT) && $(PY) -m werewolf_harness.scripts.compat_check --db --fingerprint
lint:
	$(PY) -m ruff check app tests/integration scripts loadtest faults
bench:
	cd $(ROOT) && $(PY) -m werewolf_harness.scripts.bench_runtime && $(PY) -m werewolf_harness.scripts.bench_llm \
	  && $(PY) -m werewolf_harness.scripts.measure_prompt_cache \
	  && $(PY) -m werewolf_harness.scripts.bench_streaming
chaos:          ## kill -9 a node mid-game and watch the other one adopt it
	cd $(ROOT) && $(PY) -m werewolf_harness.scripts.chaos_demo
faults:         ## fault injection: real processes, every scenario, scenario x invariant matrix -> faults/report.md
	cd $(ROOT) && $(PY) -m werewolf_harness.faults run $(FAULTS_ARGS)
faults-quick:   ## the CI subset (< 90 s); in CI Postgres/Redis are shared services, restarts are emulated
	cd $(ROOT) && $(PY) -m werewolf_harness.faults run --quick --out $(or $(FAULTS_OUT),werewolf_harness/faults/report-quick.md) $(FAULTS_ARGS)
load:           ## WebSocket load test against a running server
	cd $(ROOT) && $(PY) -m werewolf_harness.loadtest.bots --tables 200
LOCK = uv pip compile pyproject.toml --universal --python-version 3.10 --generate-hashes --no-header --no-annotate --quiet
lock:           ## re-pin requirements{,-dev}.lock (all platforms, hashes) after editing pyproject.toml
	{ sed -n '1,4p' requirements.lock; $(LOCK); } > requirements.lock.new && mv requirements.lock.new requirements.lock
	{ sed -n '1,2p' requirements-dev.lock; $(LOCK) --extra dev; } > requirements-dev.lock.new \
	  && mv requirements-dev.lock.new requirements-dev.lock

# ------------------------------------------------------------------ containers
up:             ## build + start the whole stack, return once every healthcheck is green
	$(COMPOSE) up -d $(BUILD) --wait $(if $(WORKERS),--scale worker=$(WORKERS),) $(SERVICES)
down:
	$(COMPOSE) down
down-v:         ## also drop the Postgres / Redis volumes
	$(COMPOSE) down -v
ps:
	$(COMPOSE) ps
logs:
	$(COMPOSE) logs -f --tail=100
smoke: up       ## up, one mock game through Caddy, metrics + scrape targets + dashboard + replay
	$(COMPOSE) exec -T api1 python -m werewolf_harness.scripts.smoke $(SMOKE_ARGS)
	@web=$$($(COMPOSE) port caddy 80 | sed 's/0.0.0.0/localhost/'); \
	 api=$$($(COMPOSE) port api1 8402 | sed 's/127.0.0.1/localhost/'); \
	 prom=$$($(COMPOSE) port prometheus 9090 2>/dev/null | sed 's/127.0.0.1/localhost/'); \
	 graf=$$($(COMPOSE) port grafana 3000 2>/dev/null | sed 's/127.0.0.1/localhost/'); \
	 curl -fsS -o /dev/null "http://$$web/healthz" && echo "  通过  宿主机 -> Caddy ($$web)"; \
	 echo "对局大厅   http://$$web/"; \
	 echo "API 直连   http://$$api/docs"; \
	 [ -n "$$prom" ] && echo "Prometheus http://$$prom/targets"; \
	 [ -n "$$graf" ] && echo "Grafana    http://$$graf/d/werewolf"; true
