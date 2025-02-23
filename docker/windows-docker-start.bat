@echo off
call windows-docker-env.bat

echo ==============================
echo === Starting the container ===
echo ==============================
if "%USE_GPU%" == "true" (
    docker compose -p %USER_NAME% up --force-recreate --remove-orphans
) else (
    docker compose -f docker-compose-no-gpu.yml -p %USER_NAME% up --force-recreate --remove-orphans
)
