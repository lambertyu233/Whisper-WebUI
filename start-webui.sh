#!/bin/bash

export HF_ENDPOINT="https://hf-mirror.com"
source venv/bin/activate
python app.py "$@"

echo "launching the app"
