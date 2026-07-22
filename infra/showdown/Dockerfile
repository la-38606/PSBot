FROM node:22-bookworm-slim

ARG SHOWDOWN_COMMIT

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt
RUN git clone https://github.com/smogon/pokemon-showdown.git \
    && cd pokemon-showdown \
    && git checkout "${SHOWDOWN_COMMIT}" \
    && npm ci \
    && cp config/config-example.js config/config.js

WORKDIR /opt/pokemon-showdown
EXPOSE 8000

HEALTHCHECK --interval=5s --timeout=3s --retries=20 \
  CMD node -e "const s=require('net').connect(8000,'127.0.0.1');s.on('connect',()=>process.exit(0));s.on('error',()=>process.exit(1))"

CMD ["node", "pokemon-showdown", "start", "--no-security"]
