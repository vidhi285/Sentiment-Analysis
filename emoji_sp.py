"""
============================================================
 Processes emoji image dataset using Apache Spark and
 generates metadata CSV containing image paths and labels.

 Tasks Performed:
 - Configure Java environment for Spark
 - Initialize Spark Session
 - Read emoji images from sentiment folders
 - Extract image paths and labels
 - Create Spark DataFrame
 - Display dataset information
 - Generate metadata CSV using Pandas
 - Stop Spark session after processing

 Sentiment Categories:
 positive
 negative
 neutral

 Output File:
 emoji_metadata.csv
============================================================
"""

import os
# Force Java 17
os.environ["JAVA_HOME"] = r"C:\Users\HP\AppData\Local\Programs\Eclipse Adoptium\jdk-17.0.18.8-hotspot" # Update this path if your Java installation is different
os.environ["PATH"] = os.environ["JAVA_HOME"] + r"\bin;" + os.environ["PATH"] # Ensure Java is in the system PATH
from pyspark.sql import SparkSession
import pandas as pd
spark = SparkSession.builder.appName("EmojiBigDataProcessing").getOrCreate() # Initialize Spark Session
dataset_path = r"C:\Users\HP\Documents\Sentiment Analysis Project\dataset\emoji_processed"
image_data = [] # To store tuples of (image_path, label) for all images in the dataset
for sentiment in ["positive", "negative", "neutral"]: # we go inside folder of each sentiment category
    folder_path = os.path.join(dataset_path, sentiment)
    for file in os.listdir(folder_path): # Loop through each file in the sentiment folder
        if file.endswith((".png", ".jpg", ".jpeg")): 
            full_path = os.path.join(folder_path, file) # Get the full path to the image file
            image_data.append((full_path, sentiment)) # We extract features like file path and label from raw data.
df = spark.createDataFrame(image_data, ["path", "label"]) # Create a Spark DataFrame from the list of image paths and labels
print(" Total Images:", df.count())
df.show(5, truncate=False) 
metadata_df = pd.DataFrame(image_data, columns=["image_path", "label"]) # Create a Pandas DataFrame for easier manipulation and saving to CSV
metadata_path = r"C:\Users\HP\Documents\Sentiment Analysis Project\emoji_metadata.csv"
metadata_df.to_csv(metadata_path, index=False)
print("\n Metadata saved at:", metadata_path)
spark.stop()
print("\n Spark Processing Completed Successfully!")