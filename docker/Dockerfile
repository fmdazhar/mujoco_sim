FROM nvcr.io/nvidia/pytorch:24.03-py3

### Sudo and basic OS packages
RUN apt update &&\
    apt-get install sudo -y

ARG PROJECT_NAME
RUN sudo DEBIAN_FRONTEND=noninteractive apt update  &&\
    # apt Dependencies
    sudo DEBIAN_FRONTEND=noninteractive apt install -y nano \
    git &&\
    # Disable error of dubious git ownership
    git config --global --add safe.directory /${PROJECT_NAME} &&\
    # apt-get Dependencies
    sudo DEBIAN_FRONTEND=noninteractive apt-get install htop -y \
    curl \
    software-properties-common \
    net-tools \
    unzip \
    wget 

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG C.UTF-8

# Install virtual display
RUN sudo apt-get update &&\
    # Package dependencies
    sudo apt-get install x11-apps -y &&\
    sudo apt-get update && \
    sudo apt-get install libx11-6 libxext6 libxau6 libxxf86vm1 libxdmcp6 libxcb1 libxrender1 libxrandr2 libxfixes3 libxi6 libxinerama1 libxkbcommon-x11-0 libgl1-mesa-glx libgl1-mesa-dri mesa-utils -y &&\
    sudo apt-get install x11-xserver-utils -y &&\
    # command "xeyes" for testing if X11 connection working
    sudo apt-get install x11-apps -y &&\
    sudo apt-get install libx11-6 libxext6 libxau6 libxxf86vm1 libxdmcp6 libxcb1 libxrender1 libxrandr2 libxfixes3 libxi6 libxinerama1 libxkbcommon-x11-0 libgl1-mesa-glx libgl1-mesa-dri mesa-utils -y &&\
    sudo apt-get install x11-xserver-utils -y &&\
    # Virtual display packages
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    novnc \
    supervisor \
    x11vnc \
    xvfb \
    xfce4 &&\
    # Firefox
    sudo apt-get install -y software-properties-common && \
    yes '' | sudo add-apt-repository ppa:mozillateam/ppa &&\
    echo 'Package: *\nPin: release o=LP-PPA-mozillateam\nPin-Priority: 1001\n' | sudo tee /etc/apt/preferences.d/mozilla-firefox &&\
    sudo apt install firefox -y &&\
    # Bug fixes
    sudo mkdir /tmp/.X11-unix &&\
    # Manual install for virtual desktop update (xfce v4.18) 
    sudo DEBIAN_FRONTEND=noninteractive add-apt-repository ppa:xubuntu-dev/staging &&\
    sudo apt install awesome -y &&\
    sudo apt-get install xterm &&\
    sudo apt-get install python3-tk -y &&\
    sudo apt update &&\
    sudo apt upgrade -y

# pretraining relevant packages
# swig for gymnasium[box2d]
RUN sudo apt-get install swig -y


# Python packages
RUN pip install gymnasium-robotics==1.2.4 &&\
    pip install seaborn==0.12.2 &&\
    pip install pytest==7.3.1 &&\
    pip install scikit-learn &&\
    pip install dm_env==1.6 mujoco==2.3.3 gymnasium==0.29.1 dm-robotics-transformations==0.8.1 imageio[ffmpeg]==2.5 lxml==5.3.0 pynput==1.7.7 hidapi==0.14.0.post4 pyyaml==6.0.2 &&\ 
    # maybe upgrade gymnasium to pip install gymnasium==1.0.0
    # TODO pip3 needed?
    pip3 install black &&\
    pip install stable-baselines3[extra] &&\
    pip install sb3_contrib==2.3.0 &&\
    pip install moviepy==1.0.3 &&\
    # Downgrade to prevent stable-baselines3 error: AttributeError: module 'cv2.dnn' has no attribute 'DictValue'
    pip install opencv-python==4.8.0.74 &&\
    pip install tensordict==0.3.0 &&\
    pip install wonderwords==2.2.0 &&\
    pip install huggingface-sb3==3.0 &&\
    pip install tbparse &&\
    # Gym only for legacy hugging face checkpoints
    pip install gym==0.26.2 &&\
    pip install optuna==4.2.1 &&\
    pip install plotly==6.0.0 &&\
    pip install kaleido==0.2.1 &&\
    # Disable outdated Pip warning
    python -m pip install --upgrade pip 


# Define ARG for build-time variables
ARG USER_NAME
ARG GROUP_NAME
ARG USER_ID
ARG PROJECT_NAME

# Set ENV if these need to be available at runtime
ENV USER_NAME=${USER_NAME}
ENV GROUP_NAME=${GROUP_NAME}
ENV USER_ID=${USER_ID}
ENV PROJECT_NAME=${PROJECT_NAME}

RUN sudo addgroup --gid ${USER_ID} ${GROUP_NAME} && sudo adduser --uid ${USER_ID} --gid ${USER_ID} ${USER_NAME} && sudo usermod -p '' ${USER_NAME} && sudo adduser ${USER_NAME} sudo &&\
    echo 'export PATH="$PATH:/home/${USER_NAME}/.local/bin"' >> ~/.bashrc  &&\
    echo "export PS1='\u@\$(id -gn):\w $ '" >> /home/${USER_NAME}/.bashrc 
USER ${USER_NAME}

ARG PROJECT_NAME
COPY --chmod=777 ./entrypoint.sh /opt/nvidia/nvidia_entrypoint.sh
# RUN sudo chmod 777 /home/${USER_NAME}/entrypoint.sh 

# Update the Python PATH in .bashrc, relevant for tensorboard
RUN echo 'export PATH=$PATH:$HOME/.local/bin' >> $HOME/.bashrc

WORKDIR /${PROJECT_NAME}

#CMD ["/home/${USER_NAME}/entrypoint.sh"]