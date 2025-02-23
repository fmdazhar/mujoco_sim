#!/bin/bash

# Default values can be changed manually during runtime.
export GROUP_NAME="docker"                                         
export PROJECT_NAME="mujoco_state"
export DOCKER_MEMORY="30g"
export DOCKER_NUM_CPUS="10.0"
export DOCKER_SHARED_MEMORY="40g"
export VNC_PORT="2302"
export VNC_PW="ipa"
export USE_GPU=false                                              # or false for without GPU
export OS=linux
# Dynamic values
export USER_NAME=${USER}                                           # Linux username
export USER_ID=$(id -u)                                            # Linux User ID  
# export GPU_ID=$(nvidia-smi -L | grep -oP '(?<=UUID: ).*?(?=\))')   # Dynamically gets the GPU UUID
# export GPU_ID=GPU-30ee2b75-d4ff-0643-6357-5716ba2663c0
export GPU_ID=GPU-13135b7d-aead-e437-2753-0bc97fdbea2c
export HOST_IP=$(hostname -I | awk '{print $1}')                   # Dynamically get the host IP


# Overwrite default parameters with flags e.g: ./docker-build.sh -GPU_ID=1234 -VNC_PORT=3
while [ $# -gt 0 ]; do
    case "$1" in
        -USER_NAME=*)
            USER_NAME="${1#*=}"
            ;;
        -USER_ID=*)
            USER_ID="${1#*=}"
            ;;
        -GROUP_NAME=*)
            GROUP_NAME="${1#*=}"
            ;;
        -PROJECT_NAME=*)
            PROJECT_NAME="${1#*=}"
            ;;
        -DOCKER_MEMORY=*)
            DOCKER_MEMORY="${1#*=}"
            ;;
        -DOCKER_NUM_CPUS=*)
            DOCKER_NUM_CPUS="${1#*=}"
            ;;
        -DOCKER_SHARED_MEMORY=*)
            DOCKER_SHARED_MEMORY="${1#*=}"
            ;;
        -VNC_PORT=*)
            VNC_PORT="${1#*=}"
            ;;
        -GPU_ID=*)
            GPU_ID="${1#*=}"
            ;;
        -VNC_PW=*)
            VNC_PW="${1#*=}"
            ;;
        -HOST_IP=*)
            HOST_IP="${1#*=}"
            ;;
        -USE_GPU=*)
            USE_GPU="${1#*=}"
            ;;
        *)
            echo "Faulty flag: $1"
            exit 1
    esac
    shift
done


echo "========================"
echo "=== User Information ==="
echo "========================"

echo "USER_NAME: $USER_NAME"
echo "USER_ID: $USER_ID"
echo "GROUP_NAME: $GROUP_NAME"
echo "PROJECT_NAME: $PROJECT_NAME"
echo "DOCKER_MEMORY: $DOCKER_MEMORY"
echo "DOCKER_NUM_CPUS: $DOCKER_NUM_CPUS"
echo "DOCKER_SHARED_MEMORY: $DOCKER_SHARED_MEMORY"
echo "HOST_IP: $HOST_IP"
echo "VNC_PORT: $VNC_PORT"
echo "GPU_ID: $GPU_ID"
echo "USE_GPU: $USE_GPU"
echo "OS: $OS"


