source .env

# Break if $CONTAINER_NAME is not set
if [ -z "$CONTAINER_NAME" ]; then
    echo "CONTAINER_NAME is not set"
    exit 1
else
    echo "CONTAINER_NAME is set to $CONTAINER_NAME"
fi

sudo docker rm -fv "$CONTAINER_NAME" || true
sudo docker compose up --build --remove-orphans -d --force-recreate $CONTAINER_NAME