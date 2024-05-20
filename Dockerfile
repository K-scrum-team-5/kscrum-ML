# syntax=docker/dockerfile:1.4
FROM python:3.10-slim AS base

WORKDIR /app

# 필요한 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libc-dev \
    git \
    bash \
    && rm -rf /var/lib/apt/lists/*

FROM base AS builder

COPY requirements.txt /app
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

COPY . /app

ENTRYPOINT ["python3"]
CMD ["app.py"]

FROM base as dev-envs

# 개발 환경 설정
# Docker CLI 도구 설치가 필요한 경우, 해당 방법을 개선합니다.
# 예시로, Docker 공식 이미지에서 바이너리를 복사하는 대신 적절한 설치 스크립트를 사용하세요.
