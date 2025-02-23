@echo off
REM Default values, can be changed manually during runtime.
set GROUP_NAME=docker
set PROJECT_NAME=demonstration-learning
set DOCKER_MEMORY=5g
set DOCKER_NUM_CPUS=3.0
set DOCKER_SHARED_MEMORY=5g
set VNC_PORT=2200
set VNC_PW=ipa
set USE_GPU=false
set HOST_IP=127.0.0.1
set USER_ID=1000
set OS=windows
set ENTRYPOINT_FILE_PATH="./"
REM Dynamic values
set USER_NAME=win-user

REM Check if NVIDIA GPU is available
nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo NVIDIA GPU found.
    REM Set USER_ID (can be adapted to your environment)
    @REM for /f "tokens=*" %%i in ('id -u') do set USER_ID=%%i
    REM Set GPU_ID by extracting the GPU UUID
    for /f "tokens=*" %%i in ('nvidia-smi -L ^| findstr /R "UUID: .*"') do set GPU_ID=%%i
    REM Set HOST_IP (can be adapted to your environment)
    @REM for /f "tokens=*" %%i in ('hostname -I') do set HOST_IP=%%i
) 
else (
    echo No NVIDIA GPU found.
)

REM Display the variables
echo ========================================
echo === User Information ===
echo ========================================
echo USER_NAME: %USER_NAME%
echo USER_ID: %USER_ID%
echo GROUP_NAME: %GROUP_NAME%
echo PROJECT_NAME: %PROJECT_NAME%
echo DOCKER_MEMORY: %DOCKER_MEMORY%
echo DOCKER_NUM_CPUS: %DOCKER_NUM_CPUS%
echo DOCKER_SHARED_MEMORY: %DOCKER_SHARED_MEMORY%
echo HOST_IP: %HOST_IP%
echo VNC_PORT: %VNC_PORT%
echo GPU_ID: %GPU_ID%
echo USE_GPU: %USE_GPU%
echo OS: %OS%
