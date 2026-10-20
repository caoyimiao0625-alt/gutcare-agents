FROM python:3.12-slim

# PDF 文本提取使用 pypdf；扫描页渲染使用 Poppler。
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    GCA_HOST=0.0.0.0 \
    GCA_PORT=8100 \
    GCA_MODE=replay

WORKDIR /app

COPY requirements.txt ./
RUN apt-get update && apt-get install -y --no-install-recommends poppler-utils \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir -r requirements.txt

COPY gutcare_agents ./gutcare_agents
COPY knowledge ./knowledge
COPY prompts ./prompts
COPY console ./console
COPY eval ./eval
COPY tools ./tools
COPY docs ./docs
COPY tests ./tests
COPY replays ./replays
COPY README.md ./

# 运行期数据（任务库、缓存）放在可挂载目录
RUN mkdir -p /app/data

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/healthz', timeout=2).status==200 else 1)"

CMD ["python", "-m", "gutcare_agents", "serve"]
