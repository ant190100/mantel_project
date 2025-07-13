#!/bin/bash

# Titanic ML Interpretation Toolbox - Quick Demo Setup
# This script sets up and runs the demo application

echo "Titanic ML Interpretation Toolbox Setup"
echo "======================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is required but not installed."
    echo "Please install Python 3.8+ and try again."
    exit 1
fi

echo "Python 3 found: $(python3 --version)"

# Install dependencies
echo "Installing dependencies..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo "Dependencies installed successfully"

# Check for .env file
if [ ! -f ".env" ]; then
    echo "No .env file found"
    echo "Creating sample .env file..."
    echo "# Add your HuggingFace API key here" > .env
    echo "HF_API_KEY=your_huggingface_api_key_here" >> .env
    echo "Please edit .env and add your HuggingFace API key for LLM explanations"
    echo "You can get a free API key at: https://huggingface.co/settings/tokens"
fi

echo "Starting the demo application..."
echo "The app will open in your browser at http://localhost:8501"
echo "Press Ctrl+C to stop the demo"
echo ""

# Run the Streamlit app
streamlit run streamlit_shap_demo.py
