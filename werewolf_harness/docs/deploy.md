# 容器部署与 CI

一台机器起整套服务：两个 API 节点（主 + 热备）、N 个模型 worker、Postgres、Redis、Caddy、Prometheus、Grafana（Jaeger 可选）。

## 命令

在 `werewolf_harness/` 目录下：

```
cp .env.example .env      # 可选；不建 .env 也能跑（mock / sim 模型，不需要 key）
make up                   # 构建镜像 + 启动 + 等所有 healthcheck 变绿再返回
make smoke                # = make up，然后打一局 mock，逐项检查，最后打印各个地址
make down                 # 停掉（数据卷保留）；make down-v 连数据一起删
make up WORKERS=4         # worker 扩到 4 个（或在 .env 里写 WORKERS=4）
make logs / make ps
docker compose -f deploy/docker-compose.yml --profile tracing up -d   # 加 Jaeger；同时设 OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:4318
```

默认端口（都可以在 .env 里改）：

| 服务 | 地址 | 说明 |
|---|---|---|
| 大厅（经 Caddy） | http://localhost:8080 | `HTTP_PORT`；VPS 上设成 80/443，并设 `DOMAIN` 走自动 HTTPS |
| api1 直连 | http://localhost:8402/docs | 只绑 127.0.0.1，调试用 |
| Prometheus | http://localhost:9090/targets | |
| Grafana | http://localhost:3000/d/werewolf | 匿名只读，看板自动加载 |

## 启动顺序

```
postgres (healthy) ─┬─> migrate (一次性：alembic upgrade head，退出码 0) ─┬─> api1, api2 (healthy) ──> caddy (healthy)
redis    (healthy) ─┘                                                     └─> worker × N (healthy: :9100/metrics)
prometheus (healthy) ──> grafana (healthy)
```

- 迁移是单独的一次性服务：API 节点启动时自己的 upgrade 发现已经在 head，什么也不做，两个节点不会同时跑 DDL。
- healthcheck 都用 `start_interval: 1s`：启动阶段每秒探一次，起来之后每 10 秒一次，所以 `--wait` 不会白等一个周期。
- Caddy 用 `lb_policy first`：流量全给 api1，api1 的 `/readyz` 不通就切到 api2。

## 镜像

`deploy/Dockerfile`，两阶段：

1. `deps`：`python:3.12-slim` 上建 venv，`pip install --require-hashes -r requirements.lock`，然后卸掉 pip。
2. 运行层：同一个 slim 基础镜像 + venv + 源码。源码属 root（只读），进程以 uid 10001 的 `app` 用户运行，唯一可写目录 `/app/data`。HEALTHCHECK 用标准库探 `/healthz`（slim 里没有 curl）。

依赖锁：

- `requirements.lock`：运行时依赖，带哈希，`uv pip compile --universal` 生成，一份文件覆盖 linux/amd64、linux/arm64（Apple Silicon）和 macOS，Python ≥ 3.10；版本钉在测试实际跑过的那一组。
- `requirements-dev.lock`：运行时 + `[dev]`，CI 和本地测试 venv 用。
- 改了 `pyproject.toml` 之后跑 `make lock`（需要 uv）。`tests/test_deploy.py` 会检查：代码里 import 的每个第三方包都在 pyproject 里声明了，每个运行时依赖都在锁文件里且带哈希。

多架构：基础镜像都是官方多架构镜像（python、postgres、redis、caddy、prometheus、grafana、jaeger 都有 arm64），Dockerfile 里没有 `--platform`，锁里每个带 C 扩展的包（asyncpg、uvloop、pydantic-core、httptools、greenlet……）都有 arm64 wheel，Mac 上原生构建，不走模拟。CI 里额外用 QEMU 构建一次 linux/arm64 来兜底。

## claude-cli 座位只能在宿主机上跑

`claude-cli` / `claude-cli:sonnet` 等座位就是一次 `claude -p` 子进程，用的是宿主机上 Claude Code 的登录状态。容器里没有 `claude`，也**不应该**把宿主机的登录凭据挂进容器。所以：

- 容器里能用的模型：`mock`、`script`、`sim*`，以及在 `.env` 里配了 key 的 API 模型。容器里 `/api/meta` 的模型列表不会出现 `claude-cli*`（已验证），建局时选了也会被 400 拒掉。
- 要用 claude-cli 座位：在宿主机上直接跑服务，`python -m werewolf_harness.cli serve`（或 `make dev` 起 Postgres/Redis，再 `make run`）。

## CI（仓库根目录 `.github/workflows/ci.yml`）

| job | 内容 |
|---|---|
| `unit` | Python 3.10 / 3.12 矩阵；按 `requirements-dev.lock` 装依赖（pip 缓存）；`make lint`（ruff）+ `make test-core` |
| `integration` | 同一矩阵；`postgres:16`、`redis:7` 作为 service；`make test-int`；之后依次尝试 `make compat`、`make faults-quick`（故障注入的 CI 子集，ADR 0013）、`make regress`——目标存在才跑，不存在就跳过 |
| `docker` | buildx 构建（GHA 层缓存）→ `make smoke BUILD=--no-build`（整套 compose，含 Grafana）→ 失败时打印 compose 日志 → QEMU 构建 linux/arm64 |
| `image` / `deploy` | 只在 main 上：推送 amd64+arm64 镜像到 ghcr；配置了 `DEPLOY_HOST` 才滚动重启 VPS（先 migrate，再 worker、api2、api1，每步 `--wait`） |

## 本次验证了什么

在开发沙箱里（Docker 29.4 / compose v5.1，x86_64）实际跑过的：

- **镜像构建**：`deploy/Dockerfile` 构建成功；运行用户 `uid=10001(app)`。应用自己的层 ≈ 107 MB（venv 106 MB + 源码 1 MB）；加上官方 `python:3.12-slim`（约 120 MB）预计整镜像 ≈ 230 MB 左右（沙箱里基础镜像是替身，所以总大小不具参考性，见下）。依赖层缓存命中后重建只需几秒。
- **整套启动**：`migrate api1 api2 worker×2/×3 postgres redis caddy prometheus` 全部 healthy；`up --wait`（不含构建）全新数据卷 ≈ 9.6 s，已有数据卷 ≈ 8.1 s；单个 API 容器重启到 `/readyz` 200 ≈ 1.9 s。
- **`make smoke` 全部通过**：经 Caddy 登录、建 8 人全 mock 局并打完（33 回合、约 1 s、0 次规则兜底）；API 与 worker 的 `/metrics`；Prometheus 抓取目标全部 up（api1、api2、每个 worker 副本经 DNS 服务发现）；Prometheus 能查到对局指标；每个模型回合都经 Redis Streams 由 worker 执行，按 worker 分别计数；`verify_replay` 重放已结束对局，结局一致（0 不一致）；宿主机 → Caddy 端口映射通。只用 `.env` 配置（端口、`WORKERS=3`）时同样通过。
- **故障切换**：`docker stop` api1 后经 Caddy 的 `/readyz` 返回 `node-2`，api1 回来后切回 `node-1`。
- **CI 命令**：在全新的 3.10 和 3.12 venv 里按 `requirements-dev.lock` 安装，`make lint`、`make test-core`（223 passed）、`make test-int`（46 passed）都通过；compat/faults/regress 目标不存在时正确跳过。workflow 用 actionlint 1.7.12 检查无报错。

只做了静态检查、没有实际跑的：

- **Grafana**：沙箱里拉不到 grafana 镜像。只检查了：provisioning 的数据源 uid 与看板一致；看板里引用的每个指标名都是代码注册过的（`tests/test_deploy.py`），其中 15/17 个在这次运行的 Prometheus 里已经有数据（另外两个是还没触发过的带标签计数器）。`make smoke` 默认会检查看板已加载，CI 的 docker job 会实际跑到这一步。
- **基础镜像是替身**：沙箱屏蔽了所有镜像仓库，所以 `python:3.12-slim`、`postgres:16-alpine`、`redis:7-alpine`、`caddy:2-alpine`、`prom/prometheus` 用 Ubuntu 24.04 软件包在本地做了同名替身（入口和路径与官方一致）。compose 编排、healthcheck、依赖顺序、端口、卷、应用镜像本身都是真的；官方镜像上的行为差异（比如 alpine 的 `wget`、官方 postgres 入口脚本）在 Mac 上跑 `make smoke` 时才会最终确认。
- **arm64 / Apple Silicon**：只检查了基础镜像和锁里的 wheel 都有 arm64 版本；没有在 arm64 上实际构建。
- **Jaeger**（`--profile tracing`）：没跑。

## Mac（Apple Silicon）实测（2026-10-15）

机器：MacBook Pro M3 Pro（11 核，18 GB），Docker Desktop 28.1.1，compose v2.35.1，全部官方镜像，原生 arm64，不走模拟。上一节里沙箱没能验证的三项（官方基础镜像、Grafana、arm64），在这里都实际跑过了。

| 项 | 结果 |
|---|---|
| `make smoke` | **全部通过**：Caddy → `/readyz`；8 人全 mock 局打完（33 回合、3.3 s、0 次规则兜底）；API 与 worker 的 `/metrics`；Prometheus 抓取目标全部 up（api1、api2、两个 worker 副本）；Prometheus 查到对局指标；模型回合经 Redis Streams 由两个 worker 执行（17 / 16 个）；**Grafana 看板自动加载**；回放校验 0 不一致；宿主机 → Caddy 通 |
| 冷启动（拉官方镜像 + 构建应用镜像 + 起整套 + 冒烟） | 116 s |
| `make up`（不构建），全新数据卷 / 已有数据卷 | 25.9 s / 17.4 s（含 Grafana 和 Prometheus 的 healthcheck） |
| 应用镜像 `werewolf-harness:local` | arm64；`docker image ls` 显示 341 MB（解包后），压缩后 72 MB |
| 故障切换 | 停掉 api1 后，经 Caddy 的 `/readyz` 由 `node-2` 应答；api1 重新启动到 `/readyz` 200 用 7.5 s，之后切回 `node-1` |

这台机器上 8080、8402、3000 端口已经被别的程序占用，所以这次把端口都改到了 1xxxx 段：`HTTP_PORT=18080 HTTPS_PORT=18443 API_PORT=18402 PROMETHEUS_PORT=19090 GRAFANA_PORT=13000 make smoke`。

## 在 Mac（Apple Silicon）上第一次跑

```
cd werewolf_harness
make smoke                 # Docker Desktop 开着即可；第一次会拉基础镜像、装依赖，约 1–2 分钟
```

最后几行应该是 `全部通过`、`通过  宿主机 -> Caddy (localhost:8080)` 和四个地址。端口被占了就在 `.env` 里改 `HTTP_PORT` / `API_PORT` / `PROMETHEUS_PORT` / `GRAFANA_PORT`。失败时看 `make logs`。
