## Run this command if you are using Linux/Mac otherwise do it Manually...............

#!/bin/bash

echo "Setting up DocMind..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Ollama not found. Please install it from https://ollama.com"
    exit 1
fi

echo "Pulling Ollama models (this may take a while)..."
ollama pull llama3.1
ollama pull nomic-embed-text

# Create required directories
mkdir -p uploads vectorstore

echo "Setup complete! Run: streamlit run app.py"