import pandas as pd
from pymongo import MongoClient, errors
import time
import sys

# ==============================================================================
# PHASE 1: MONGODB STORAGE
# ==============================================================================
# This script efficiently reads a large CSV dataset (~3M rows) in chunks and 
# inserts it into a local MongoDB instance.
# 
# Why MongoDB for this project?
# 1. Scalability: MongoDB handles large datasets far better than loading a 
#    3GB CSV file into RAM all at once.
# 2. Fast Querying: We can index the 'sentiment' column to quickly pull 
#    balanced batches during training.
# 3. Flexible Schema: If we want to add more metadata (e.g., embeddings) 
#    later, document databases handle schema evolution gracefully.
# ==============================================================================

# --- Configuration ---
CSV_FILE_PATH = "balanced_sentiment_dataset.csv"
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "sentiment_analysis_db"
COLLECTION_NAME = "tweets_dataset"
CHUNK_SIZE = 50000  # Number of rows per batch

def connect_to_mongodb():
    """Establishes connection to MongoDB."""
    try:
        print(f"Connecting to MongoDB at {MONGO_URI}...")
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
        # Force a call to check if server is available
        client.server_info()
        print("Successfully connected to MongoDB.")
        return client
    except errors.ServerSelectionTimeoutError as err:
        print(f"Failed to connect to MongoDB. Is the server running locally?\nError: {err}")
        sys.exit(1)

def ingest_data_in_chunks(client):
    """Reads CSV in chunks and inserts into MongoDB using insert_many."""
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    # Optional: Clear existing data to avoid duplicates on re-run
    print(f"Clearing existing data in '{COLLECTION_NAME}' collection...")
    collection.delete_many({})

    # Create index for optimized querying
    print("Creating index on 'sentiment' field...")
    collection.create_index("sentiment")

    total_inserted = 0
    start_time = time.time()

    print(f"Starting data ingestion from '{CSV_FILE_PATH}'...")
    
    try:
        # Read CSV in chunks to prevent memory overflow
        for i, chunk in enumerate(pd.read_csv(CSV_FILE_PATH, chunksize=CHUNK_SIZE)):
            # Drop rows with missing values if any
            chunk = chunk.dropna(subset=['clean_text', 'sentiment'])
            
            # Convert DataFrame chunk to list of dictionaries
            records = chunk.to_dict(orient='records')
            
            if records:
                try:
                    # Batch insert
                    collection.insert_many(records, ordered=False)
                    total_inserted += len(records)
                    
                    # Progress logging
                    elapsed_time = time.time() - start_time
                    print(f"Batch {i + 1}: Inserted {len(records)} records. "
                          f"Total so far: {total_inserted}. "
                          f"Elapsed time: {elapsed_time:.2f} seconds.")
                          
                except errors.BulkWriteError as bwe:
                    print(f"Warning: Bulk write error on batch {i+1}: {bwe.details}")
                    
    except FileNotFoundError:
        print(f"Error: Could not find '{CSV_FILE_PATH}'. Please ensure the file exists.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during ingestion: {e}")
        sys.exit(1)

    print("\n" + "="*50)
    print("INGESTION COMPLETE")
    print("="*50)
    print(f"Total documents successfully inserted: {total_inserted}")
    
    # Verify count directly from MongoDB
    db_count = collection.count_documents({})
    print(f"Total documents in MongoDB collection: {db_count}")
    
    total_time = time.time() - start_time
    print(f"Total execution time: {total_time:.2f} seconds.")
    print("="*50)

if __name__ == "__main__":
    mongo_client = connect_to_mongodb()
    ingest_data_in_chunks(mongo_client)
    mongo_client.close()
