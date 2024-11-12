#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status.

# Create a directory for datasets
mkdir -p datasets
cd datasets

download_and_extract() {
    local url=$1
    local zip_file=$2
    local dataset_number=$3

    echo "Downloading dataset $dataset_number..."
    if curl -L "$url" > "$zip_file"; then
        echo "Download complete. Extracting..."
        if unzip -o "$zip_file"; then
            echo "Extraction complete."
            
            # Move and rename the data.yaml file
            if [ -f "data.yaml" ]; then
                mv "data.yaml" "data_$dataset_number.yaml"
                echo "Renamed data.yaml to data_$dataset_number.yaml"
            else
                echo "Warning: data.yaml not found in dataset $dataset_number"
            fi
            
            # Merge train, valid, and test directories
            for dir in train valid test; do
                mkdir -p "../$dir"
                if [ -d "$dir" ]; then
                    echo "Moving $dir files for dataset $dataset_number"
                    mv "$dir"/* "../$dir/" 2>/dev/null || true
                else
                    echo "Warning: $dir directory not found in dataset $dataset_number"
                fi
            done
            
            # Clean up
            rm "$zip_file"
            echo "Cleaned up temporary files for dataset $dataset_number"
        else
            echo "Error: Failed to extract $zip_file"
            return 1
        fi
    else
        echo "Error: Failed to download dataset $dataset_number"
        return 1
    fi
}

# Replace these URLs with your own Roboflow dataset URLs
# Format: https://universe.roboflow.com/ds/XXXXX?key=YYYYY
download_and_extract "YOUR_ROBOFLOW_URL_1" "roboflow1.zip" 1
download_and_extract "YOUR_ROBOFLOW_URL_2" "roboflow2.zip" 2
download_and_extract "YOUR_ROBOFLOW_URL_3" "roboflow3.zip" 3
download_and_extract "YOUR_ROBOFLOW_URL_4" "roboflow4.zip" 4
download_and_extract "YOUR_ROBOFLOW_URL_5" "roboflow5.zip" 5
download_and_extract "YOUR_ROBOFLOW_URL_6" "roboflow6.zip" 6
download_and_extract "YOUR_ROBOFLOW_URL_7" "roboflow7.zip" 7

echo "All datasets downloaded and extracted successfully."

# Count total images
total_images=$(find . -type f \( -name "*.jpg" -o -name "*.jpeg" -o -name "*.png" \) | wc -l)
echo "Total number of images: $total_images"