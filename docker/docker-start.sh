source ./docker-env.sh

echo "=============================="
echo "=== Starting the container ==="
echo "=============================="
# Check if GPU should be used
if [ "${USE_GPU}" = "true" ]; then
  docker compose -f docker-compose.yml -p ${USER}_${PROJECT_NAME}_2 up --force-recreate 
else
  docker compose -f docker-compose-no-gpu.yml -p ${USER}_${PROJECT_NAME}_2 up --force-recreate
fi