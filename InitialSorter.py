import os
import shutil

def is_mri(file_path):
    return file_path.endswith('.nii.gz') and not any(x in file_path for x in ['seg', 'mask', 'reg', 'bet'])

def is_mask(file_path):
    return 'seg' in file_path and file_path.endswith('.nii.gz')

def is_segmentation_info(file_path):
    return any(file_path.endswith(ext) for ext in ['.json', '.tfm', '.mat'])

def copy_and_rename(src_path, dest_path, criteria):
    for item in os.listdir(src_path):
        item_path = os.path.join(src_path, item)
        if os.path.isfile(item_path) and criteria(item_path):
            base_name, extension = os.path.splitext(item)  # Split the name and extension
            if extension == '.gz':  # Handle .nii.gz as a special case
                base_name, _ = os.path.splitext(base_name)
                extension = '.nii.gz'
            dest_file_path = os.path.join(dest_path, item)
            counter = 1
            while os.path.exists(dest_file_path):
                dest_file_path = os.path.join(dest_path, f"{base_name}_{counter}{extension}")
                counter += 1
            print(f"Copying {item_path} to {dest_file_path}")
            shutil.copy(item_path, dest_file_path)
        elif os.path.isdir(item_path):
            copy_and_rename(item_path, dest_path, criteria)

def process_patient_folder(patient_folder_path, dest_base_path):
    dest_mri = os.path.join(dest_base_path, 'Patient_MRIs')
    dest_masks = os.path.join(dest_base_path, 'Masks')
    dest_segmentation_info = os.path.join(dest_base_path, 'Segmentation_Info')

    os.makedirs(dest_mri, exist_ok=True)
    os.makedirs(dest_masks, exist_ok=True)
    os.makedirs(dest_segmentation_info, exist_ok=True)

    copy_and_rename(patient_folder_path, dest_mri, is_mri)
    copy_and_rename(patient_folder_path, dest_masks, is_mask)
    copy_and_rename(patient_folder_path, dest_segmentation_info, is_segmentation_info)

    print(f"Processed patient folder: {patient_folder_path}")

dest_base_path = r'C:\Users\rahhu\OneDrive\Desktop\ISEF_GBM\P'

for patient_number in range(1, 92):
    patient_folder_path = os.path.join(r'C:\Users\rahhu\OneDrive\Desktop\ISEF_GBM\Imaging', f'Patient({patient_number})')
    process_patient_folder(patient_folder_path, dest_base_path)