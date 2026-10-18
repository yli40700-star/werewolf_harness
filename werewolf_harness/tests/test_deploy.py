"""Static checks on the deployment files, so a broken wire shows in the unit job
rather than on the first `make up`:

- every third-party module the code imports is declared in pyproject.toml, and
  every runtime dependency is pinned (with hashes) in requirements.lock;
- compose: every app container waits for healthy Postgres/Redis and a finished
  migration, every long-running service has a healthcheck;
- Prometheus scrapes the services compose actually defines, on the ports they serve;
- every metric the Grafana dashboard queries is one the code registers.
"""
from __future__ import annotations

import ast
import pathlib
import re
import sys
from importlib import metadata

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
yaml = pytest.importorskip("yaml")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _req_name(req: str) -> str:
    return _norm(re.split(r"[\s\[<>=!~;]", req.strip(), maxsplit=1)[0])


def _pyproject() -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:                     # 3.10
        import tomli as tomllib
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _declared() -> tuple[set[str], set[str]]:
    p = _pyproject()["project"]
    runtime = {_req_name(r) for r in p["dependencies"]}
    extras = {_req_name(r) for reqs in p.get("optional-dependencies", {}).values() for r in reqs}
    return runtime, extras


def _third_party_imports() -> dict[str, set[str]]:
    std = set(sys.stdlib_module_names)
    out: dict[str, set[str]] = {}
    for p in ROOT.rglob("*.py"):
        rel = p.relative_to(ROOT)
        if rel.parts[0] in ("tests", ".venv", "resume"):
            continue
        for n in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
            if isinstance(n, ast.Import):
                mods = [a.name for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
                # `from opentelemetry import trace`: the name may itself be a (namespace) submodule
                mods = [f"{n.module}.{a.name}" for a in n.names]
            else:
                continue
            for m in mods:
                if m.split(".")[0] not in std and m.split(".")[0] != "werewolf_harness":
                    out.setdefault(m, set()).add(str(rel))
    return out


def _owner(mod: str, by_path: dict[str, str]) -> str | None:
    """The installed distribution whose RECORD holds this module's files.

    Walks the dotted path from the longest prefix down, so `fastapi.Depends`
    resolves through `fastapi` and `opentelemetry.trace` through its own folder
    (packages_distributions() cannot see namespace packages on 3.10)."""
    parts = mod.split(".")
    for i in range(len(parts), 0, -1):
        base = "/".join(parts[:i])
        for key in (f"{base}/__init__.py", f"{base}.py"):
            if key in by_path:
                return by_path[key]
        hit = next((d for k, d in by_path.items() if k.startswith(base + "/")), None)
        if hit:
            return hit
        hit = next((d for k, d in by_path.items() if k.startswith(base + ".") and k.endswith((".so", ".pyd"))), None)
        if hit:
            return hit
    return None


def test_every_import_is_declared():
    runtime, extras = _declared()
    declared = runtime | extras
    # one level of transitive requirements counts (starlette comes with fastapi)
    via = set()
    for d in declared:
        try:
            via |= {_req_name(r) for r in metadata.requires(d) or []}
        except metadata.PackageNotFoundError:
            pass
    by_path = {str(f): _norm(d.metadata["Name"]) for d in metadata.distributions() for f in (d.files or [])}
    missing = {}
    for mod, files in _third_party_imports().items():
        # not installed here (an optional extra such as yappi): it must be declared under its own name
        owner = _owner(mod, by_path) or _norm(mod.split(".")[0])
        if owner not in declared | via:
            missing[mod] = (owner, sorted(files))
    assert not missing, f"imported but not declared in pyproject.toml: {missing}"


def test_lock_pins_every_runtime_dependency_with_hashes():
    runtime, _ = _declared()
    text = (ROOT / "requirements.lock").read_text()
    pinned = {_norm(m) for m in re.findall(r"^([A-Za-z0-9_.\-]+)==", text, re.M)}
    assert runtime <= pinned, f"in pyproject but not in requirements.lock (run `make lock`): {runtime - pinned}"
    blocks = re.split(r"\n(?=[A-Za-z0-9])", text.split("\n", 4)[-1])
    unhashed = [b.split()[0] for b in blocks if "==" in b and "--hash=sha256:" not in b]
    assert not unhashed, unhashed


def _compose() -> dict:
    return yaml.safe_load((DEPLOY / "docker-compose.yml").read_text())


def test_compose_ordering_and_healthchecks():
    svcs = _compose()["services"]
    for name in ("api1", "api2", "worker"):
        dep = svcs[name]["depends_on"]
        assert dep["postgres"]["condition"] == "service_healthy"
        assert dep["redis"]["condition"] == "service_healthy"
        assert dep["migrate"]["condition"] == "service_completed_successfully"
    assert svcs["migrate"]["restart"] == "no"
    for name in ("postgres", "redis", "worker", "prometheus", "grafana"):
        assert "test" in svcs[name]["healthcheck"], name
    # the api nodes inherit the image's HEALTHCHECK
    assert "HEALTHCHECK" in (DEPLOY / "Dockerfile").read_text()
    assert svcs["caddy"]["depends_on"]["api1"]["condition"] == "service_healthy"


def test_dockerfile_runs_unprivileged_from_the_lock():
    df = (DEPLOY / "Dockerfile").read_text()
    assert re.search(r"^USER app$", df, re.M)
    assert "--require-hashes" in df and "requirements.lock" in df
    # multi-arch: no platform pinning, a slim official base
    assert "--platform" not in df and not re.search(r"^FROM .*amd64", df, re.M)
    assert re.search(r"^FROM python:\$\{PYTHON_VERSION\}-slim", df, re.M)


def test_prometheus_scrapes_what_compose_runs():
    svcs = _compose()["services"]
    prom = yaml.safe_load((DEPLOY / "prometheus.yml").read_text())
    for job in prom["scrape_configs"]:
        for sc in job.get("static_configs", []):
            for t in sc["targets"]:
                host, port = t.split(":")
                assert host in svcs and port == "8402", t
        for sd in job.get("dns_sd_configs", []):
            for n in sd["names"]:
                assert n in svcs
                assert str(sd["port"]) == svcs[n]["environment"]["WORKER_METRICS_PORT"]


def test_dashboard_queries_registered_metrics():
    import json

    from prometheus_client import REGISTRY

    from werewolf_harness.app.obs import metrics  # noqa: F401  -- registers the collectors
    families = {m.name for m in REGISTRY.collect()}
    board = json.loads((DEPLOY / "grafana" / "dashboards" / "werewolf.json").read_text())
    used = {n for p in board["panels"] for t in p.get("targets", []) for n in re.findall(r"wh_[a-z_]+", t["expr"])}
    assert used
    unknown = {n for n in used if n not in families and re.sub(r"_(bucket|count|sum|total)$", "", n) not in families}
    assert not unknown, unknown
    ds = yaml.safe_load((DEPLOY / "grafana" / "provisioning" / "datasources" / "prom.yml").read_text())
    uids = {d["uid"] for d in ds["datasources"]}
    assert {p["datasource"]["uid"] for p in board["panels"]} <= uids
