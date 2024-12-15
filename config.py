import os
import shutil

# Define the paths
original_images_path = "Original Images/FOLDS"  # Path to the original dataset
new_dataset_path = "dataset"  # Path to the new dataset structure

# Create the new dataset directory if it doesn't exist
os.makedirs(new_dataset_path, exist_ok=True)

# Define the categories
categories = ["chickenpox", "cowpox", "healthy", "hfmd", "measles", "monkeypox"]

# Create folders for each category in the new dataset
for category in categories:
    os.makedirs(os.path.join(new_dataset_path, category), exist_ok=True)

# Traverse through the original dataset
for fold in os.listdir(original_images_path):
    fold_path = os.path.join(original_images_path, fold)
    
    # Check if it is a directory
    if os.path.isdir(fold_path):
        for split in ["Train", "Test", "Valid"]:
            split_path = os.path.join(fold_path, split)
            
            # Check if the split directory exists
            if os.path.isdir(split_path):
                for category in categories:
                    category_path = os.path.join(split_path, category)
                    
                    # Check if the category directory exists
                    if os.path.isdir(category_path):
                        # Copy all images from the category to the new dataset folder
                        for image_name in os.listdir(category_path):
                            image_path = os.path.join(category_path, image_name)
                            # Only copy files (skip if it's a directory)
                            if os.path.isfile(image_path):
                                shutil.copy(image_path, os.path.join(new_dataset_path, category, image_name))

print("Dataset has been reorganized successfully!")