"""
============================================================
 Resizes all emoji images in the dataset to 48x48 pixels
 and verifies image dimensions after processing.

 Tasks Performed:
 - Traverse all folders inside dataset
 - Detect image files (.png, .jpg, .jpeg)
 - Resize every image to 48x48
 - Save resized images
 - Verify image dimensions
 - Track incorrectly resized images
 - Display processing summary

 Output:
 - All images converted to 48x48 resolution
 - Displays total processed images
 - Shows any failed/wrong-sized images
============================================================
"""


import os
from PIL import Image # For image processing

dataset_path = r"C:\Users\HP\Documents\Sentiment Analysis Project\dataset\emoji_processed"

processed = 0
wrong_images = [] # To track any images that are not 48x48

for root, dirs, files in os.walk(dataset_path): # Walk through all files in the dataset directory
    for file in files: # Check if the file is an image (you can adjust this based on your dataset)
        if file.endswith((".png", ".jpg", ".jpeg")):
            path = os.path.join(root, file) # Get full path to the image

            try:
                img = Image.open(path) # Open the image using PIL (Python Imaging Library)

                # Resize
                img = img.resize((48, 48))
                img.save(path)
                # Verify
                if img.size != (48, 48):
                    wrong_images.append(file)

                processed += 1
            except Exception as e:
                print("Error:", file)

print(f" Total images processed: {processed}")
if not wrong_images:
    print(" All images are 48x48")
else:
    print("Some images are not 48x48:")
    for img in wrong_images:
        print(img)