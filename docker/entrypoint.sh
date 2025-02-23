#!/bin/bash
# Copyright (c) 2016-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Gather parts in alpha order
shopt -s nullglob extglob
_SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"
declare -a _PARTS=(
  $( ( for i in "${_SCRIPT_DIR}/entrypoint.d"/*@(.txt|.sh) /nvidia/*/entrypoint.d/*@(.txt|.sh); do
         echo "$(basename "$i")" "$i"
       done
     ) | sort | awk '{print $2}'
  )
)
shopt -u nullglob extglob

# Execute the entrypoint parts
for _file in "${_PARTS[@]}"; do
  case "${_file}" in
    *.txt) cat "${_file}";;
    *.sh)  source "${_file}";;
  esac
done

echo

# This script can either be a wrapper around arbitrary command lines,
# or it will simply exec bash if no arguments were given
# if [[ $# -eq 0 ]]; then
#   exec "/bin/bash"
# else
#   exec "$@"
# fi

echo "=================="
echo "== Docker setup =="
echo "=================="
echo
# Start virtual desktop in the background
nohup /usr/bin/Xvfb :0 -screen 0 1280x720x24 > /tmp/Xvfb.log 2>&1 &\
nohup /usr/bin/x11vnc -display :0 -nopw -listen localhost -xkb -forever -shared -noxdamage -noshm -passwd ${VNC_PW} > /tmp/x11vnc.log 2>&1 &\
nohup /usr/share/novnc/utils/launch.sh --vnc localhost:5900 > /tmp/novnc.log 2>&1 &\
nohup awesome > /tmp/awesome.log 2>&1 &

# Remove dekarb_nesting.egg-info
sudo rm -rf ${PROJECT_NAME}.egg-info

# Install my_project repo for imports
pip install -e .

#echo "${USER_NAME}:${PW}" | sudo chpasswd # Change the user sudo password
#unset PW # Remove PW from Env variable

# Info message for the user
echo
echo "==================="
echo "=== Information ==="
echo "==================="
echo
echo "Open a display via a web browser with the following URL:"
echo
echo "  http://${HOST_IP}:${VNC_PORT}/vnc.html "
echo
echo "Attach to the container via a new terminal and the command:"
echo
echo "  docker exec -it ${USER_NAME}_${PROJECT_NAME} bash "
echo
echo "  OR "
echo
echo "  Use Vscode 'Dev Containers' Extension to attach Vscode to the container. "
echo
echo "Shut down the container using this terminal with: Ctrl + C"
echo
echo "##############################################"
# Let the container run as a server
wait
