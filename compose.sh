source .env

# Break if $CONTAINER_NAME is not set
if [ -z "$CONTAINER_NAME" ]; then
    echo "CONTAINER_NAME is not set"
    exit 1
else
    echo "CONTAINER_NAME is set to $CONTAINER_NAME"
fi

# Check if mcp/cookies.txt exists
if [ ! -f "mcp/cookies.txt" ]; then
    echo "Error: mcp/cookies.txt file not found"
    exit 1
else
    echo "mcp/cookies.txt file found"
fi

sudo docker rm -fv "$CONTAINER_NAME" || true
sudo docker compose up --build --remove-orphans -d --force-recreate $CONTAINER_NAME