#!/bin/bash

# Dataset Download Script for SimCLR Semantic Segmentation

echo "Setting up data directories..."
mkdir -p experiments/data/{images,annotations}

# Download Oxford-IIIT Pet Dataset
echo "Downloading Oxford-IIIT Pet Dataset..."
cd experiments/data

# Download images
if [ ! -f "images.tar.gz" ]; then
    echo "Downloading images..."
    wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
fi

# Download annotations
if [ ! -f "annotations.tar.gz" ]; then
    echo "Downloading annotations..."
    wget http://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
fi

# Extract files
echo "Extracting files..."
tar -xzf images.tar.gz
tar -xzf annotations.tar.gz

# Clean up problematic files
echo "Cleaning dataset..."
cd annotations/trimaps
find . -type f -name '._*.png' -exec rm {} + 2>/dev/null || true

cd ../../images
find . -type f -name '*.mat' -exec rm {} + 2>/dev/null || true

cd ../..

echo "Dataset download and setup complete!"
echo "Data located in: experiments/data/"