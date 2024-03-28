import os
import numpy as np
import nibabel as nib
from skimage.transform import resize
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate, Conv3DTranspose, BatchNormalization, Activation, ZeroPadding3D, Cropping3D, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import psutil
import matplotlib.pyplot as plt

class NiftiDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_filenames, mask_filenames, batch_size, target_size=(256, 256, 32)):
        self.image_filenames, self.mask_filenames = image_filenames, mask_filenames
        self.batch_size = batch_size
        self.target_size = target_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))
    
    def __getitem__(self, idx):
        batch_x_paths = self.image_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y_paths = self.mask_filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        # Use ThreadPoolExecutor to load files
        with ThreadPoolExecutor(max_workers=12) as executor:
            batch_x = list(executor.map(NiftiDataGenerator.load_nifti_file, batch_x_paths, [self.target_size] * len(batch_x_paths)))
            batch_y = list(executor.map(NiftiDataGenerator.load_nifti_file, batch_y_paths, [self.target_size] * len(batch_y_paths)))

        return np.array(batch_x), np.array(batch_y)
    
    @staticmethod
    def load_nifti_file(file_path, target_size=(256, 256, 32)):
        try:
            nifti = nib.load(file_path)
            image = nifti.get_fdata(dtype=np.float32)
            resized_image = resize(image, target_size, mode='constant', anti_aliasing=True)
            resized_image = resized_image[..., np.newaxis]  # Add channel dimension
            print(f"Loaded file {file_path}: Shape {resized_image.shape}")
            return resized_image
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
        return np.zeros(target_size + (1,), dtype=np.float32)

def preprocess_data(base_dir, mri_types, mask_types):
    images, masks = [], []

    for mri_type, mask_type in zip(mri_types, mask_types):
        mri_dir = os.path.join(base_dir, mri_type)
        mask_dir = os.path.join(base_dir, mask_type)

        mri_files = [f for f in os.listdir(mri_dir) if f.endswith('.nii.gz')]
        mask_files = [f for f in os.listdir(mask_dir) if f.endswith('.nii.gz')]

        for mri_file in mri_files:
            corresponding_mask = mri_file.replace(mri_type, mask_type)
            if corresponding_mask in mask_files:
                full_image_path = os.path.join(mri_dir, mri_file)
                full_mask_path = os.path.join(mask_dir, corresponding_mask)
                images.append(full_image_path)
                masks.append(full_mask_path)

    return images, masks

def adjust_tensor(tensor_to_adjust, target_tensor):
    # Calculate the difference in each dimension
    diff = [target_tensor.shape[i] - tensor_to_adjust.shape[i] for i in range(1, 4)]

    # Apply padding or cropping as needed
    if any(d < 0 for d in diff):  # Needs cropping
        cropping = [(-d//2, -d//2 + (-d % 2)) if d < 0 else (0, 0) for d in diff]
        tensor_to_adjust = Cropping3D(cropping=cropping)(tensor_to_adjust)
    elif any(d > 0 for d in diff):  # Needs padding
        padding = [(d//2, d//2 + (d % 2)) if d > 0 else (0, 0) for d in diff]
        tensor_to_adjust = ZeroPadding3D(padding=padding)(tensor_to_adjust)

    return tensor_to_adjust
def crop_and_concat(up_tensor, down_tensor):
    # Calculating the difference in size
    diff_depth = down_tensor.shape[3] - up_tensor.shape[3]
    diff_height = down_tensor.shape[1] - up_tensor.shape[1]
    diff_width = down_tensor.shape[2] - up_tensor.shape[2]

    # Cropping the down_tensor before concatenation
    down_tensor_cropped = Cropping3D(cropping=((diff_height // 2, diff_height - diff_height // 2),
                                               (diff_width // 2, diff_width - diff_width // 2),
                                               (diff_depth // 2, diff_depth - diff_depth // 2)))(down_tensor)

    return concatenate([up_tensor, down_tensor_cropped], axis=4)
def unet_model(input_size=(256, 256, 32, 1)):
    inputs = Input(input_size)

    # Downsampling Path
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    # Bottleneck
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv3D(512, (3, 3, 3), activation='relu', padding='same')(conv5)

    # Upsampling Path with adjusted concatenation
    up6 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(conv5)
    up6 = adjust_tensor(up6, conv4)
    up6 = concatenate([up6, conv4], axis=4)
    up6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    up6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)

    up7 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(up6)
    up7 = adjust_tensor(up7, conv3)
    up7 = concatenate([up7, conv3], axis=4)
    up7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    up7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)

    up8 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(up7)
    up8 = adjust_tensor(up8, conv2)
    up8 = concatenate([up8, conv2], axis=4)
    up8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    up8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)

    up9 = Conv3DTranspose(32, (2, 2, 2), strides=(2, 2, 2), padding='same')(up8)
    up9 = adjust_tensor(up9, conv1)
    up9 = concatenate([up9, conv1], axis=4)
    up9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)
    up9 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(up9)

    # Final Convolutional Layer
    final_layer = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')(up9)

    model = Model(inputs=inputs, outputs=final_layer)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    return model


#def unet_model(input_size=(256, 256, 32, 1)):
    inputs = Input(input_size)

    # Downsampling Path
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(inputs)
    #conv1 = BatchNormalization()(conv1)
    conv1 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv1)
    #conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling3D(pool_size=(2, 2, 2))(conv1)

    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool1)
    #conv2 = BatchNormalization()(conv2)
    conv2 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv2)
    #conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling3D(pool_size=(2, 2, 2))(conv2)

    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(pool2)
    #conv3 = BatchNormalization()(conv3)
    conv3 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(conv3)
    #conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling3D(pool_size=(2, 2, 2))(conv3)

    # Removed conv4 and conv5 layers to reduce depth

    # Upsampling Path with skip connection
    up6 = Conv3DTranspose(256, (2, 2, 2), strides=(2, 2, 2), padding='same')(pool3)
    up6 = crop_and_concat(up6, conv3)
    up6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    #up6 = BatchNormalization()(up6)
    up6 = Conv3D(256, (3, 3, 3), activation='relu', padding='same')(up6)
    #up6 = BatchNormalization()(up6)

    up7 = Conv3DTranspose(128, (2, 2, 2), strides=(2, 2, 2), padding='same')(up6)
    up7 = crop_and_concat(up7, conv2)
    up7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    #up7 = BatchNormalization()(up7)
    up7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(up7)
    #up7 = BatchNormalization()(up7)

    up8 = Conv3DTranspose(64, (2, 2, 2), strides=(2, 2, 2), padding='same')(up7)
    up8 = crop_and_concat(up8, conv1)
    up8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    #up8 = BatchNormalization()(up8)
    up8 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(up8)
    #up8 = BatchNormalization()(up8)

    # Final Convolutional Layer
    final_layer = Conv3D(1, (1, 1, 1), activation='sigmoid', padding='same')(up8)

    model = Model(inputs=inputs, outputs=final_layer)
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

    return model

base_dir = r'C:\Users\rahhu\OneDrive\Desktop\ISEF_GBM\P\mom'
print("Starting data preprocessing...")
def plot_results(train_losses, train_accuracies, val_losses, val_accuracies):
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.show()
def save_model_h5(model, save_path):
    """
    Save the entire model to a HDF5 file.
    
    Parameters:
    model (tf.keras.Model): The Keras model to save.
    save_path (str): The path to save the model file.
                     This should include the name of the file and .h5 extension.
    """
    model.save(save_path, save_format='h5')
    print(f"Model saved to {save_path}")

# Call the plot_results function with the tracked metrics

def train_and_evaluate_model(model, train_gen, val_gen, epochs=9):
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    for epoch in range(epochs):
        print(f"\033[92mEpoch {epoch + 1}/{epochs}\033[0m, RAM Usage: \033[94m{psutil.virtual_memory().percent}%\033[0m")
        train_loss, train_acc, val_loss, val_acc = 0, 0, 0, 0
        num_train_batches, num_val_batches = len(train_gen), len(val_gen)

        # Training loop with updated tqdm usage
        pbar = tqdm(total=num_train_batches, desc=f"\033[96mTraining Epoch {epoch + 1}\033[0m", ncols=100, colour='blue')
        for batch_idx, (image_batch, mask_batch) in enumerate(train_gen):
            slice_loss, slice_acc = model.train_on_batch(image_batch, mask_batch)
            train_loss += slice_loss
            train_acc += slice_acc
            pbar.set_postfix_str(f"Batch {batch_idx + 1}/{num_train_batches}, Loss: {slice_loss:.4f}, Acc: {slice_acc:.4f}")
            pbar.update(1)
        pbar.close()

        # Validation loop with updated tqdm usage
        pbar = tqdm(total=num_val_batches, desc=f"\033[93mValidating Epoch {epoch + 1}\033[0m", ncols=100, colour='green')
        for val_batch_idx, (val_image_batch, val_mask_batch) in enumerate(val_gen):
            slice_loss, slice_acc = model.evaluate(val_image_batch, val_mask_batch, verbose=0)
            val_loss += slice_loss
            val_acc += slice_acc
            pbar.set_postfix_str(f"Batch {val_batch_idx + 1}/{num_val_batches}, Loss: {slice_loss:.4f}, Acc: {slice_acc:.4f}")
            pbar.update(1)
        pbar.close()

        # Calculate and log the results
        train_loss /= num_train_batches
        train_acc /= num_train_batches
        val_loss /= num_val_batches
        val_acc /= num_val_batches
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)
        print(f"\033[95mEpoch {epoch+1} Results\033[0m: Train Loss = \033[91m{train_loss}\033[0m, Train Accuracy = \033[92m{train_acc}\033[0m, Val Loss = \033[93m{val_loss}\033[0m, Val Accuracy = \033[94m{val_acc}\033[0m")

    return train_losses, train_accuracies, val_losses, val_accuracies


# repare data
base_dir = r'C:\Users\rahhu\OneDrive\Desktop\ISEF_GBM\P\mom'
mri_types = ['CT1', 'FLAIR', 'T1', 'T2']
mask_types = ['ct1_seg_mask', 'flair_seg_mask', 't1_seg_mask', 't2_seg_mask']
images, masks = preprocess_data(base_dir, mri_types, mask_types)
train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=42)
# Create data generators
train_gen = NiftiDataGenerator(train_images, train_masks, batch_size=1)
val_gen = NiftiDataGenerator(test_images, test_masks, batch_size=1)

# Setup model and callbacks
model = unet_model()
checkpoint = ModelCheckpoint('unet_model.h5', verbose=1, save_best_only=True)
early_stopping = EarlyStopping(patience=10, verbose=1)

# Train model
train_losses, train_accuracies, val_losses, val_accuracies = train_and_evaluate_model(model, train_gen, val_gen)

# Visualization
plot_results(train_losses, train_accuracies, val_losses, val_accuracies)
# After your model has been trained and you have the `model` object:
save_model_h5(model, 'C:\\Users\\rahhu\\OneDrive\\Desktop\\ISEF_GBM\\P\\mom\\your_model.h5')
