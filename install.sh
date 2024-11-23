#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Variables
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER"
INSTALL_DIR="$PWD/miniconda"
ENV_FILE="environment.yaml"

# Step 1: Download Miniconda installer
echo "Downloading Miniconda installer..."
curl -fsSL -o "$MINICONDA_INSTALLER" "$MINICONDA_URL"

# Step 2: Install Miniconda locally
echo "Installing Miniconda in $INSTALL_DIR..."
bash "$MINICONDA_INSTALLER" -b -p "$INSTALL_DIR"

# Step 3: Initialize Miniconda
echo "Initializing Miniconda..."
eval "$($INSTALL_DIR/bin/conda shell.bash hook)"

# Step 4: Create environment from YAML file
if [ -f "$ENV_FILE" ]; then
    echo "Creating environment from $ENV_FILE..."
    conda env create --file "$ENV_FILE"
else
    echo "Environment file $ENV_FILE not found. Exiting."
    exit 1
fi

# Step 5: Extract environment name from YAML and activate it
ENV_NAME=$(grep -E '^name:' "$ENV_FILE" | awk '{print $2}')
if [ -z "$ENV_NAME" ]; then
    echo "Environment name not found in $ENV_FILE. Exiting."
    exit 1
fi

echo "Activating environment: $ENV_NAME"
conda activate "$ENV_NAME"

# Step 6: Cleanup
echo "Cleaning up installer..."
rm -f "$MINICONDA_INSTALLER"

echo "Setup complete. Miniconda installed and environment activated."