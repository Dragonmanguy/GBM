import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
import tensorflow as tf
import matplotlib.pyplot as plt

# Function to load and preprocess MRI data
def preprocess_mri(file_path, target_size=(256, 256, 32)):
    nifti = nib.load(file_path)
    image = nifti.get_fdata(dtype=np.float32)
    resized_image = resize(image, target_size, mode='constant', anti_aliasing=True)
    resized_image = resized_image[..., np.newaxis]  # Add channel dimension
    return resized_image

# Load the trained model
model_path = 'C:\\Users\\rahhu\\OneDrive\\Desktop\\ISEF_GBM\\P\\usable_MRIs\\your_model.h5'
model = tf.keras.models.load_model(model_path)

# Preprocess the input MRI
input_mri_path = r'C:\Users\rahhu\OneDrive\Desktop\ISEF_GBM\P\usable_MRIs\CT1\CT1_15.nii.gz'  # Replace with your MRI file path
input_mri = preprocess_mri(input_mri_path)

# Model prediction
predicted_mask = model.predict(np.expand_dims(input_mri, axis=0))

# Post-processing to binary mask
threshold = 0.3
binary_mask = (predicted_mask > threshold).astype(np.uint8)

# Function to overlay mask on MRI
def overlay_mask_on_mri(mri, mask):
    red_mask = np.zeros_like(mri[..., 0], dtype=np.uint8)  # Creating a 3D mask
    red_mask[mask == 0] = 255  # Mark non-tumor regions

    # Overlay red mask onto original MRI
    overlayed_image = np.copy(mri[..., 0])  # Take the first channel for overlay
    overlayed_image = np.stack([overlayed_image]*3, axis=-1)  # Convert to 3-channel image
    overlayed_image[..., 0][mask == 0] = red_mask[mask == 0]  # Apply red color to non-tumor regions

    return overlayed_image

# Apply the mask overlay
overlayed_mri = overlay_mask_on_mri(input_mri, binary_mask[0, ..., 0])

# Save the overlayed images
output_dir = 'C:\\Users\\rahhu\\OneDrive\\Desktop\\ISEF_GBM\\Slices_Saved'
os.makedirs(output_dir, exist_ok=True)

# Save each slice as a separate image
for i in range(overlayed_mri.shape[2]):
    slice_img = overlayed_mri[:, :, i]

    # Normalize the slice image values to be within 0 to 1
    slice_img_normalized = slice_img / 255.0

    # Ensure the values are within the 0..1 range
    slice_img_normalized = np.clip(slice_img_normalized, 0, 1)

    plt.imsave(os.path.join(output_dir, f'slice_{i}.png'), slice_img_normalized)

print("All slices saved to the folder 'pushinP'.")