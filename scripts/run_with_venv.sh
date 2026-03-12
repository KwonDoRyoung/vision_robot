#!/bin/bash
# 패키지 경로 기준으로 venv 활성화 후 Python 스크립트 실행
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PKG_DIR/venv/bin/python3"

if [ ! -f "$VENV_PYTHON" ]; then
    echo "[ERROR] venv not found at $PKG_DIR/venv"
    echo "Run: python3 -m venv --system-site-packages $PKG_DIR/venv"
    exit 1
fi

exec "$VENV_PYTHON" "$@"
