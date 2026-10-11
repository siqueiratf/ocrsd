FROM nginxinc/nginx-unprivileged:1.28-alpine
RUN sed -i 's/worker_processes  auto;/worker_processes  2;/' /etc/nginx/nginx.conf
COPY deploy/nginx.conf /etc/nginx/conf.d/default.conf
