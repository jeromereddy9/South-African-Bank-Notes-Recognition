import os
import re

def validate_dataset_filenames(file_path, pattern):
   
    total_files = 0
    valid_files = 0
    invalid_names = []

    # Loop through all files in the directory
    for filename in os.listdir(file_path):
        total_files += 1
        
        # Check if filename matches the expected pattern
        if re.search(pattern, filename):
            valid_files += 1
        else:
            invalid_names.append(filename)  # Collect invalid names for reporting

    # Report results
    if total_files == valid_files:
        print("All files names valid.")
        return True
    else:
        print(f"Invalid file names detected: \n{invalid_names}")
        return False