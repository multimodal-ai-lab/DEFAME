FROM debian:bookworm-slim

MAINTAINER Mark Rothermel "mark.rothermel@tu-darmstadt.de"

WORKDIR /

# Install and update basic services
RUN apt update

# Install Firecrawl dependencies (Node.js, pnpm, redis)
RUN apt -y install curl && \
    curl -sL https://deb.nodesource.com/setup_20.x | bash - && \
    apt -y install nodejs
RUN npm install pnpm --global
RUN apt -y install redis-server

# Install Firecrawl
RUN apt -y install git
RUN git clone https://github.com/mendableai/firecrawl.git
WORKDIR /firecrawl/apps/api
RUN pnpm install
COPY .env .env

WORKDIR /firecrawl

EXPOSE 3002

# Execute Firecrawl
CMD cd /firecrawl/apps/api && \
    { redis-server & pnpm run workers & pnpm run start; }
