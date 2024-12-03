#!/bin/bash

# Variables
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_URL="https://repo.anaconda.com/miniconda/$MINICONDA_INSTALLER"
INSTALL_DIR="$PWD/miniconda"
ENV_FILE="environment.yaml"

# Step 0: Check if env file exists and extract env name from it
if [ ! -f $ENV_FILE ]; then
  echo "Environment file does not exist."
  return 1
fi

ENV_NAME=$(grep -E '^name:' "$ENV_FILE" | awk '{print $2}')
if [ -z $ENV_NAME ]; then
  echo "Environment name not found in $ENV_FILE."
  return 1
fi

# Step 1: Download and install Miniconda
echo "Downloading and installing Miniconda..."
if [ ! -d $INSTALL_DIR ]; then
  wget "$MINICONDA_URL" -O "$MINICONDA_INSTALLER"
  bash "$MINICONDA_INSTALLER" -b -p "$INSTALL_DIR"
else
  echo "Miniconda already installed. For reinstallation delete: $INSTALL_DIR and rerun this script."
fi

# Step 3: Initialize Miniconda
echo "Initializing Miniconda..."
source miniconda/bin/activate

# Step 4: Create environment from YAML file
echo "Creating environment from $ENV_FILE..."
if ! conda env list | grep -q "^$ENV_NAME "; then
  conda env create --file "$ENV_FILE"
else
  echo "Conda environment already exists."
fi

# Step 5: Activate conda env
echo "Activating environment: $ENV_NAME..."
conda activate "$ENV_NAME"

# Step 6: Cleanup
if [ -f $MINICONDA_INSTALLER ]; then
  echo "Cleaning up installer..."
  rm -f "$MINICONDA_INSTALLER"
fi

echo "Setup complete."
echo "For automatic conda activation run command:"
echo "\t \$conda init --all"
