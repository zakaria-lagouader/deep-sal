#!/bin/bash

# Set the folder containing .ply files
ply_folder="./dataset-3"

# Check if the folder exists
if [ ! -d "$ply_folder" ]; then
  echo "Error: Folder $ply_folder does not exist."
  exit 1
fi

# Iterate over each .ply file in the folder
for ply_file in "$ply_folder"/*.ply; do
  if [ -f "$ply_file" ]; then
    # Extract file name without extension
    base_name=$(basename "$ply_file" .ply)
    
    # Define paths
    obj_file="$ply_file"
    txt_file="$ply_folder/$base_name.face.txt"
    output_folder="./data2"
    n_number="10000"

    # Call the executable
    ./decimate_sal.exe "$obj_file" "$txt_file" "$output_folder" "$n_number"
    
    # Optionally, you can add some feedback
    echo "Processed: $ply_file"
  fi
done
