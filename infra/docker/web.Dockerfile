# OvaScope web (Next.js standalone output).
FROM node:22.20.0-alpine AS deps
WORKDIR /web
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM node:22.20.0-alpine AS build
WORKDIR /web
ENV NEXT_TELEMETRY_DISABLED=1 NEXT_OUTPUT=standalone
COPY --from=deps /web/node_modules ./node_modules
COPY apps/web ./
RUN npm run build

FROM node:22.20.0-alpine AS run
WORKDIR /web
ENV NODE_ENV=production NEXT_TELEMETRY_DISABLED=1 PORT=3000 HOSTNAME=0.0.0.0
RUN addgroup -S ovascope && adduser -S ovascope -G ovascope
COPY --from=build /web/.next/standalone ./
COPY --from=build /web/.next/static ./.next/static
USER ovascope
EXPOSE 3000
CMD ["node", "server.js"]
