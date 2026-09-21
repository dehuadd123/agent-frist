# 企业级 LLM 智能应用平台 - 零依赖部署镜像
FROM python:3.12-alpine

WORKDIR /app

# 复制全部源码与内置语料
COPY . /app

# 端口可通过环境变量覆盖，默认 8000
ENV PORT=8000
EXPOSE 8000

# 直接以标准库 http.server 启动，无需安装任何依赖
CMD ["python", "server.py"]
