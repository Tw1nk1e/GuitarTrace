@echo off
chcp 65001 >nul
title GuitarTrace
cd /d "%~dp0"

where uv >nul 2>nul
if errorlevel 1 (
  echo Не найден uv. Установите: https://docs.astral.sh/uv/getting-started/installation/
  pause
  exit /b 1
)

if not exist "models\birefnet-lite.onnx" goto download_models
if exist "models\sam2.1_hiera_tiny.pt" goto launch

:download_models
  echo Первый запуск: скачиваю модели, ~380 МБ...
  uv run python scripts/fetch_models.py
  if errorlevel 1 (
    pause
    exit /b 1
  )

:launch
echo Запускаю сервис... (браузер откроется сам; закройте это окно, чтобы остановить)
uv run python -m gbcv.server --open
pause
