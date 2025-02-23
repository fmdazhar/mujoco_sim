@echo off
call windows-docker-env.bat

echo ======================================
echo === Starting docker build process ===
echo ======================================
if "%USE_GPU%" == "true" (
    docker compose build
) else (
    docker compose -f docker-compose-no-gpu.yml build
)
