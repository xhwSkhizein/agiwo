FROM node:20-bookworm-slim AS web-builder

WORKDIR /build/console/web
COPY console/web/package.json console/web/package-lock.json ./
RUN npm ci
COPY console/web/ ./
RUN npm run build

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEXT_TELEMETRY_DISABLED=1

WORKDIR /opt/agiwo

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        jq \
        nginx \
        nodejs \
        npm \
        ripgrep \
        wget \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir yt-dlp

RUN groupadd --system agiwo \
    && useradd --system --gid agiwo --create-home --home-dir /home/agiwo agiwo

COPY pyproject.toml README.md ./
COPY agiwo ./agiwo
RUN pip install --no-cache-dir .

COPY console/pyproject.toml ./console/pyproject.toml
COPY console/server ./console/server
RUN pip install --no-cache-dir ./console

RUN mkdir -p /opt/agiwo-console/web /data /var/log/nginx /tmp/nginx \
    && chown -R agiwo:agiwo /opt/agiwo-console/web /data /var/log/nginx /tmp/nginx /home/agiwo

COPY --from=web-builder /build/console/web/.next /opt/agiwo-console/web/.next
COPY --from=web-builder /build/console/web/node_modules /opt/agiwo-console/web/node_modules
COPY --from=web-builder /build/console/web/package.json /opt/agiwo-console/web/package.json
COPY --from=web-builder /build/console/web/package-lock.json /opt/agiwo-console/web/package-lock.json
COPY --from=web-builder /build/console/web/public /opt/agiwo-console/web/public
COPY --from=web-builder /build/console/web/next.config.ts /opt/agiwo-console/web/next.config.ts

COPY console/docker/entrypoint.sh /usr/local/bin/agiwo-console-entrypoint.sh
COPY console/docker/healthcheck.sh /usr/local/bin/agiwo-console-healthcheck.sh
COPY console/docker/nginx.conf /etc/nginx/nginx.conf
RUN chmod +x /usr/local/bin/agiwo-console-entrypoint.sh /usr/local/bin/agiwo-console-healthcheck.sh \
    && chown -R agiwo:agiwo /opt/agiwo-console/web /usr/local/bin/agiwo-console-entrypoint.sh /usr/local/bin/agiwo-console-healthcheck.sh

EXPOSE 8422
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=6 CMD ["/usr/local/bin/agiwo-console-healthcheck.sh"]

USER agiwo

ENTRYPOINT ["/usr/local/bin/agiwo-console-entrypoint.sh"]
