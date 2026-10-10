FROM node:22-bookworm-slim AS build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci
COPY frontend ./
RUN npm run build
FROM nginxinc/nginx-unprivileged:1.28-alpine
RUN sed -i 's/worker_processes  auto;/worker_processes  2;/' /etc/nginx/nginx.conf
COPY --from=build /app/dist /usr/share/nginx/html
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
