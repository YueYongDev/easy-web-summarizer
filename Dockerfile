# 用轻量 Python 基础镜像
FROM python:3.12-slim

# 基本环境
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # 默认把容器内的 Ollama 地址指向宿主机（Mac/Windows可用；Linux请看下面 run 命令）
    OLLAMA_BASE_URL=http://host.docker.internal:11434 \
    OLLAMA_MODEL=qwen3:4b

# 可选：装个 curl 用于健康检查/排错
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先拷依赖，加快构建缓存利用
COPY requirements.txt .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 拷代码（假设你的文件名是 web_summarizer_api.py）
COPY . .

# 暴露端口（你的服务是 8001）
EXPOSE 8001

# 直接启动应用（0.0.0.0 让容器外可访问）
CMD ["python", "app/web_summarizer_api.py", "--host", "0.0.0.0", "--port", "8001"]