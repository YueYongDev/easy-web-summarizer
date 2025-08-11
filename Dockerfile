# 用轻量 Python 基础镜像
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    OLLAMA_MODEL=qwen3:4b \
    # 关键：让 transformers 不去 import torchvision（可留可删）
    TRANSFORMERS_NO_TORCHVISION=1

# 安装 Chromium 以及 ChromeDriver 运行所需依赖
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    chromium \
    ca-certificates \
    fonts-liberation \
    libasound2 \
    libatk-bridge2.0-0 libatk1.0-0 \
    libnss3 libxss1 libxrandr2 libxdamage1 libxfixes3 \
    libxkbcommon0 libxcomposite1 libxrender1 libxext6 libx11-6 libxcb1 libx11-xcb1 \
    libdrm2 libgbm1 libglib2.0-0 libgtk-3-0 libu2f-udev libvulkan1 \
    unzip gnupg dumb-init curl \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先拷依赖，加快构建缓存利用
COPY requirements.txt .

# 安装 Python 依赖（确保 selenium 在里面）
# 如果 requirements.txt 还没有，请加：selenium>=4.21.0
RUN pip install --no-cache-dir -r requirements.txt

# 拷代码
COPY . .

# 暴露端口
EXPOSE 8001

# 用 dumb-init 保证信号转发干净
ENTRYPOINT ["dumb-init", "--"]
CMD ["python", "app/web_summarizer_api.py", "--host", "0.0.0.0", "--port", "8001"]