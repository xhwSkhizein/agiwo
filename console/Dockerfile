FROM node:20-bookworm-slim AS web-builder

ENV NEXT_TELEMETRY_DISABLED=1 \
    NPM_CONFIG_UPDATE_NOTIFIER=false

WORKDIR /build/console/web
COPY console/web/package.json console/web/package-lock.json ./
RUN --mount=type=cache,target=/root/.npm npm ci
COPY console/web/ ./
RUN npm run build

FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEXT_TELEMETRY_DISABLED=1 \
    NPM_CONFIG_UPDATE_NOTIFIER=false \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    HOME=/tmp/agiwo-home

WORKDIR /opt/agiwo

ARG DEBIAN_FRONTEND=noninteractive

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    rm -f /etc/apt/apt.conf.d/docker-clean \
    && sed -i 's|http://deb.debian.org|https://deb.debian.org|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get -o Acquire::Retries=5 \
        -o Acquire::http::Timeout=30 \
        -o Acquire::https::Timeout=30 \
        update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        jq \
        nginx \
        ripgrep \
        wget

COPY --from=web-builder /usr/local/bin/node /usr/local/bin/node
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked <<'EOF'
set -e
if /usr/local/bin/node --version; then
    exit 0
fi

apt-get -o Acquire::Retries=5 \
    -o Acquire::http::Timeout=30 \
    -o Acquire::https::Timeout=30 \
    update
apt-get install -y --no-install-recommends libc6 libgcc-s1 libstdc++6
/usr/local/bin/node --version
EOF

COPY pyproject.toml ./
RUN --mount=type=cache,target=/root/.cache/pip <<'EOF'
set -e
python - <<'PY' > /tmp/agiwo-requirements.txt
import tomllib

with open("pyproject.toml", "rb") as pyproject:
    project = tomllib.load(pyproject)["project"]

for dependency in project["dependencies"]:
    print(dependency)
PY
PIP_CACHE_DIR=/root/.cache/pip pip install hatchling yt-dlp -r /tmp/agiwo-requirements.txt
EOF

COPY README.md ./
COPY agiwo ./agiwo
COPY templates ./templates
RUN --mount=type=cache,target=/root/.cache/pip PIP_CACHE_DIR=/root/.cache/pip pip install --no-deps --no-build-isolation .

COPY console/docker/browser-cli-wheels /tmp/browser-cli-wheels
RUN --mount=type=cache,target=/root/.cache/pip <<'EOF'
set -e
wheels="$(find /tmp/browser-cli-wheels -maxdepth 1 -type f -name '*.whl' | sort)"
wheel_count="$(printf '%s\n' "$wheels" | sed '/^$/d' | wc -l)"
if [ "$wheel_count" -gt 1 ]; then
    echo "Expected at most one Browser CLI wheel, found $wheel_count:" >&2
    printf '%s\n' "$wheels" >&2
    exit 1
fi
if [ "$wheel_count" -eq 1 ]; then
    wheel="$wheels"
    PIP_CACHE_DIR=/root/.cache/pip pip install "$wheel"
fi
rm -rf /tmp/browser-cli-wheels
EOF

COPY console/pyproject.toml ./console/pyproject.toml
RUN --mount=type=cache,target=/root/.cache/pip <<'EOF'
set -e
python - <<'PY' > /tmp/agiwo-console-requirements.txt
import re
import tomllib

with open("console/pyproject.toml", "rb") as pyproject:
    project = tomllib.load(pyproject)["project"]

for dependency in project["dependencies"]:
    package = re.split(r"\s|[<>=~!]", dependency, maxsplit=1)[0].split("[", maxsplit=1)[0]
    if package == "agiwo":
        continue
    print(dependency)
PY
PIP_CACHE_DIR=/root/.cache/pip pip install -r /tmp/agiwo-console-requirements.txt
EOF

COPY console/server ./console/server
COPY console/Dockerfile ./console/Dockerfile
COPY console/docker ./console/docker
RUN rm -f ./console/docker/browser-cli-wheels/*.whl
RUN --mount=type=cache,target=/root/.cache/pip PIP_CACHE_DIR=/root/.cache/pip pip install --no-deps --no-build-isolation ./console

RUN groupadd --system agiwo \
    && useradd --system --gid agiwo --create-home --home-dir /home/agiwo agiwo

RUN install -d -o agiwo -g agiwo /opt/agiwo-console/web /data /tmp/nginx /tmp/agiwo-home /home/agiwo \
    && install -d -m 0777 /var/log/nginx \
    && chmod 1777 /tmp/nginx /tmp/agiwo-home \
    && chmod 0777 /var/log/nginx

RUN install -d -m 0777 /data/root/skills \
    && chown -R agiwo:agiwo /data/root

COPY --chown=agiwo:agiwo --from=web-builder /build/console/web/.next/standalone /opt/agiwo-console/web
COPY --chown=agiwo:agiwo --from=web-builder /build/console/web/.next/static /opt/agiwo-console/web/.next/static
COPY --chown=agiwo:agiwo --from=web-builder /build/console/web/public /opt/agiwo-console/web/public

COPY --chown=agiwo:agiwo console/docker/entrypoint.sh /usr/local/bin/agiwo-console-entrypoint.sh
COPY --chown=agiwo:agiwo console/docker/healthcheck.sh /usr/local/bin/agiwo-console-healthcheck.sh
COPY console/docker/nginx.conf /etc/nginx/nginx.conf
RUN chmod +x /usr/local/bin/agiwo-console-entrypoint.sh /usr/local/bin/agiwo-console-healthcheck.sh

EXPOSE 8422
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 CMD ["/usr/local/bin/agiwo-console-healthcheck.sh"]

USER agiwo

ENTRYPOINT ["/usr/local/bin/agiwo-console-entrypoint.sh"]
