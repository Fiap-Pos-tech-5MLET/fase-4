FROM python:3.13.3-slim

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    libffi-dev \
    libpq-dev \
    libssl-dev \
    nginx \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copiar requirements e instalar dependências Python
COPY requirements.txt .
ENV PIP_ROOT_USER_ACTION=ignore
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código da aplicação
COPY . .

# Definir variável de ambiente para produção
ENV ENVIRONMENT=production

# Copiar arquivo de configuração do Nginx
COPY nginx.conf /etc/nginx/sites-available/default

# Copiar página de landing
COPY index.html /app/index.html

# Criar arquivo de configuração do supervisor para rodar múltiplos processos
RUN mkdir -p /var/log/supervisor
COPY <<EOF /etc/supervisor/conf.d/supervisord.conf
[supervisord]
nodaemon=true
logfile=/var/log/supervisor/supervisord.log
user=root

[program:nginx]
command=/usr/sbin/nginx -g "daemon off;"
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/nginx.err.log
stdout_logfile=/var/log/supervisor/nginx.out.log

[program:api]
command=uvicorn app.main:app --host 127.0.0.1 --port 8000
directory=/app
autostart=true
autorestart=true
stderr_logfile=/var/log/supervisor/api.err.log
stdout_logfile=/var/log/supervisor/api.out.log
EOF

# Expor porta 80 (Nginx)
EXPOSE 80

# Rodar supervisor que gerencia todos os 3 processos
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
