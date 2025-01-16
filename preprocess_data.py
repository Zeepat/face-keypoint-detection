import os
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import logging
from tqdm import tqdm

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Configure logging
logging.basicConfig(
    filename='preprocessing.log',
    level=logging.DEBUG,  # Set to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def resize_image_and_adjust_keypoints(img_filename, original_images_dir, output_dir, annotations, idx, target_size=(224, 224)):
    """
    Resizes an image to the target size and adjusts keypoints accordingly.
    Searches for the image within subdirectories.

    Args:
        img_filename (str): Filename of the image.
        original_images_dir (str): Directory containing original images.
        output_dir (str): Directory to save the resized image.
        annotations (pd.DataFrame): DataFrame containing annotations.
        idx (int): Index of the current image in the DataFrame.
        target_size (tuple): Desired image size (height, width).

    Returns:
        list or None: Updated keypoints for the resized image or None if an error occurs.
    """
    try:
        # Walk through directories to find the image
        img_path = None
        for root, dirs, files in os.walk(original_images_dir):
            if img_filename in files:
                img_path = os.path.join(root, img_filename)
                break

        if img_path is None:
            logging.warning(f"Image {img_filename} not found in {original_images_dir}. Skipping.")
            return None

        # Open and resize the image
        with Image.open(img_path) as img:
            img = img.convert('RGB')
            original_width, original_height = img.size
            # Choose the appropriate resampling filter
            img_resized = img.resize((target_size[1], target_size[0]), Image.BILINEAR)  # Changed to BILINEAR
            resized_width, resized_height = img_resized.size
            # Save the resized image
            img_resized.save(os.path.join(output_dir, img_filename))
            logging.debug(f"Resized image {img_filename} from ({original_width}, {original_height}) to ({resized_width}, {resized_height}).")

        # Extract and validate keypoints
        keypoints = annotations.loc[idx, ['LeftEyeCenter_x', 'LeftEyeCenter_y', 
                     'RightEyeCenter_x', 'RightEyeCenter_y',
                     'NoseCenter_x', 'NoseCenter_y',  
                     'MouthCenter_x', 'MouthCenter_y']].values.astype('float')

        # Check for NaN or infinite values
        if not np.all(np.isfinite(keypoints)):
            logging.warning(f"Non-finite keypoints found for {img_filename}. Skipping.")
            return None

        # Ensure there are exactly 8 keypoints
        if keypoints.shape[0] != len(keypoints.flatten()):
            logging.warning(f"Incorrect number of keypoints for {img_filename}. Expected 8, got {keypoints.shape[0]}. Skipping.")
            return None

        # Reshape and scale keypoints
        keypoints = keypoints.reshape(-1, 2)  # Shape: (4, 2)
        keypoints[:, 0] = keypoints[:, 0] * (resized_width / original_width)
        keypoints[:, 1] = keypoints[:, 1] * (resized_height / original_height)
        updated_keypoints = keypoints.flatten().tolist()  # [x1, y1, x2, y2, ...]

        logging.debug(f"Updated keypoints for {img_filename}: {updated_keypoints}")

        return updated_keypoints

    except Exception as e:
        logging.error(f"Error processing {img_filename}: {e}")
        return None

def preprocess_dataset(original_annotations_csv, original_images_dir, 
                      processed_annotations_csv, processed_images_dir, 
                      target_size=(224, 224)):
    """
    Preprocesses the dataset by resizing images and adjusting keypoints.

    Args:
        original_annotations_csv (str): Path to the original annotations CSV.
        original_images_dir (str): Directory containing original images.
        processed_annotations_csv (str): Path to save the processed annotations CSV.
        processed_images_dir (str): Directory to save resized images.
        target_size (tuple): Desired image size (height, width).
    """
    # Create the processed images directory if it doesn't exist
    os.makedirs(processed_images_dir, exist_ok=True)

    # Load original annotations
    annotations = pd.read_csv(original_annotations_csv)
    logging.info(f"Loaded {len(annotations)} annotations from {original_annotations_csv}.")

    # Initialize a list to store processed annotations
    processed_annotations = []

    # Iterate over each image and process with a progress bar
    for idx in tqdm(range(len(annotations)), desc="Preprocessing Images"):
        img_filename = annotations.loc[idx, "file_id"]

        # Ensure filename has an extension; if not, attempt to append common extensions
        if not os.path.splitext(img_filename)[1]:
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                trial_filename = img_filename + ext
                trial_path = os.path.join(original_images_dir, trial_filename)
                if os.path.exists(trial_path):
                    img_filename = trial_filename
                    found = True
                    logging.debug(f"Appended extension {ext} to {img_filename}.")
                    break
            if not found:
                logging.warning(f"Image {img_filename} does not have a valid extension or does not exist. Skipping.")
                continue

        updated_keypoints = resize_image_and_adjust_keypoints(
            img_filename=img_filename,
            original_images_dir=original_images_dir,
            output_dir=processed_images_dir,
            annotations=annotations,
            idx=idx,
            target_size=target_size
        )

        if updated_keypoints is not None:
            # Create a new annotation row with updated keypoints
            new_row = annotations.loc[idx].copy()
            new_row[['LeftEyeCenter_x', 'LeftEyeCenter_y', 
                     'RightEyeCenter_x', 'RightEyeCenter_y',
                     'NoseCenter_x', 'NoseCenter_y',  
                     'MouthCenter_x', 'MouthCenter_y']] = updated_keypoints
            processed_annotations.append(new_row)
            logging.debug(f"Processed and appended annotations for {img_filename}.")

    # Check if any annotations were processed
    if not processed_annotations:
        logging.error("No annotations were processed. Check if images and annotations match.")
        print("No annotations were processed. Please check the logs for details.")
        return

    # Create a new DataFrame with processed annotations
    processed_df = pd.DataFrame(processed_annotations)
    logging.info(f"Processed annotations count: {len(processed_df)}.")

    # Save the processed annotations
    processed_df.to_csv(processed_annotations_csv, index=False)
    logging.info(f"Processed annotations saved to {processed_annotations_csv}.")
    print(f"Preprocessing completed. Processed data saved to {processed_annotations_csv} and {processed_images_dir}.")

if __name__ == "__main__":
    # Define paths
    ORIGINAL_ANNOTATIONS_CSV = 'data/data_all_kps/Annotations/annotations.csv'
    ORIGINAL_IMAGES_DIR = 'data/data_all_kps/Images'
    PROCESSED_ANNOTATIONS_CSV = 'data/data_all_kps/Annotations/processed_annotations.csv'
    PROCESSED_IMAGES_DIR = 'data/data_all_kps/Images_Processed'
    
    # Run preprocessing
    preprocess_dataset(
        original_annotations_csv=ORIGINAL_ANNOTATIONS_CSV,
        original_images_dir=ORIGINAL_IMAGES_DIR,
        processed_annotations_csv=PROCESSED_ANNOTATIONS_CSV,
        processed_images_dir=PROCESSED_IMAGES_DIR,
        target_size=(224, 224)
    )