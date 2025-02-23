source ./docker-env.sh
echo "====================================="
echo "=== Starting docker build process ==="
echo "====================================="
# Check if GPU should be used
if [ "${USE_GPU}" = "true" ]; then
  docker compose -f docker-compose.yml build
else
  docker compose -f docker-compose-no-gpu.yml build
fi
