#!/bin/sh
exec uvicorn youtube:app --host 0.0.0.0 --port "${PORT:-5000}"
