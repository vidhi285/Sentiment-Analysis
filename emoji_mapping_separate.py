"""
============================================================
 emoji_csv_generator.py
 Reads emoji metadata CSV and creates separate CSV files
 for each platform (Apple, Samsung, Google, Facebook, Windows)

 Dataset Structure:
 - Reads image paths from emoji folders
 - Matches emoji IDs with sentiment labels
 - Generates platform-wise CSV datasets

 Labels:
 positive / negative / neutral

 Output Files:
 apple_emoji.csv
 samsung_emoji.csv
 google_emoji.csv
 facebook_emoji.csv
 windows_emoji.csv
============================================================
"""

import os
import pandas as pd
import re
base_path = r"C:\Users\HP\Documents\Sentiment Analysis Project\dataset\emojis\image"
metadata_path = r"C:\Users\HP\Documents\Sentiment Analysis Project\emoji_metadata.csv"
metadata_df = pd.read_csv(metadata_path)
print("Columns:", metadata_df.columns)
def extract_id(path): #Function to extract number from path
    match = re.search(r'\d+', str(path)) #Finds first number in the string
    if match:
        return int(match.group())
    return None
metadata_df['emoji_id'] = metadata_df['image_path'].apply(extract_id) #Applies function to every row & Extracts ID from image_path
mapping = dict(zip(metadata_df['emoji_id'], metadata_df['label'])) #Creates a dictionary mapping emoji_id to label
print("✅ Mapping Ready")
platforms = ["Apple", "Samsung", "Google", "Facebook", "Windows"]
for platform in platforms: #loop for each platform/folder
    folder_path = os.path.join(base_path, platform)
    if not os.path.exists(folder_path):
        print(f"⚠️ {platform} not found, skipping...")
        continue
    output_file = f"{platform.lower()}_emoji.csv"
    if os.path.exists(output_file):
        print(f"⚠️ {output_file} already exists, skipping...")
        continue
    results = [] #List to store results for current platform
    for file in os.listdir(folder_path): #reads all file in the folder
        match = re.search(r'\d+', file) #extraact number from file name
        if not match:
            continue
        emoji_id = int(match.group()) #convert extracted number to integer
        label = mapping.get(emoji_id, "unknown") #Check if ID exists in mapping if found assign number and not then write unknown
        full_path = os.path.join(folder_path, file)
        results.append([
            full_path,
            label
        ])
        print(f"{platform} → {file} → {label}")
    df = pd.DataFrame(results, columns=[ #convert ist into table
        "path",
        "label"
    ])
    df.to_csv(output_file, index=False) #Save file without index column
    print(f"✅ {output_file} saved!\n")
print("🎉 ALL DONE!")

