import os
import shutil
import re

# Define the folder to search in
root_folder = r'C:\Users\rahhu\OneDrive\Desktop\ISEF_GBM\P'

# Define the new destination folder 'mom'
dest_folder = os.path.join(root_folder, 'mom')

# Create the 'mom' directory if it doesn't exist
os.makedirs(dest_folder, exist_ok=True)

# Define your different naming schemes
naming_schemes = [
    'cti_brain_mask', 'CT1_r2s_bet_reg', 'CT1_r2s_bet',
    'ct1_seg_mask', 'ct1_skull_strip', 'flair_brain_mask',
    'FLAIR_r2s_bet_reg', 'FLAIR_r2s_bet', 'flair_seg_mask',
    'flair_skull_strip', 'seg_mask', 'segmentation_CT1_origspace',
    'segmentation_FLAIR_origspace', 'segmentation_T1_origspace',
    'segmentation_T2_origspace', 'segmentation', 't1_brain_mask',
    'T1_r2s_bet_reg', 'T1_r2s_bet', 't1_seg_mask',
    't1_skull_strip', 't2_brain_mask', 'T2_r2s_bet_reg',
    'T2_r2s_bet', 't2_seg_mask', 't2_skull_strip',
    'CT1', 'FLAIR', 'T1', 'T2'
]

# Create a subdirectory for each naming scheme within 'mom'
for scheme in naming_schemes:
    os.makedirs(os.path.join(dest_folder, scheme), exist_ok=True)

# Function to determine if a file matches a naming scheme and its variant
def is_scheme_variant(filename, scheme):
    pattern = re.compile(rf"{scheme}(_\d+)?\.nii(\.gz)?$")
    return pattern.match(filename)

# Function to get the correct destination folder and handle duplicates
def get_dest_file_path(src_file, scheme):
    base_name, ext = os.path.splitext(src_file)
    if ext == '.gz':  # Special handling for .nii.gz files
        base_name, _ = os.path.splitext(base_name)
        ext = '.nii.gz'
    dest_file = os.path.join(dest_folder, scheme, os.path.basename(base_name) + ext)
    counter = 1
    while os.path.exists(dest_file):
        dest_file = os.path.join(dest_folder, scheme, f"{os.path.basename(base_name)}_{counter}{ext}")
        counter += 1
    return dest_file

# Traverse the directory and its subdirectories
for foldername, subfolders, filenames in os.walk(root_folder):
    for filename in filenames:
        src_file_path = os.path.join(foldername, filename)
        for scheme in naming_schemes:
            if is_scheme_variant(filename, scheme):
                dest_file_path = get_dest_file_path(src_file_path, scheme)
                shutil.copy(src_file_path, dest_file_path)
                print(f"Copied {filename} to {dest_file_path}.")
                break

print("Done.")