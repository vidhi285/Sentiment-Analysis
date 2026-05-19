import os
import pandas as pd

# Folder where your CSV files are stored
csv_folder = r"C:\Users\khush\OneDrive\Desktop\StudioBasedProj"

# Your correct base path for images
new_base_path = r"C:\Users\khush\OneDrive\Desktop\StudioBasedProj\image"

# Loop through all CSV files
for file in os.listdir(csv_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(csv_folder, file)

        print(f"\nProcessing: {file}")

        df = pd.read_csv(file_path)

        # Check if 'path' column exists
        if 'path' not in df.columns:
            print(f"'path' column missing in {file}, skipping...")
            continue

        updated_paths = []

        for old_path in df['path']:
            try:
                # Extract filename (example: 1001.png)
                filename = os.path.basename(old_path)

                # Extract platform (Apple, Facebook, etc.)
                platform = os.path.basename(os.path.dirname(old_path))

                # Construct new path
                new_path = os.path.join(new_base_path, platform, filename)

                updated_paths.append(new_path)

            except Exception as e:
                print(f"Error processing: {old_path}")
                updated_paths.append(old_path)

        # Replace path column
        df['path'] = updated_paths

        # Save updated copy (creates updated_apple_emoji.csv etc.)
        output_path = os.path.join(csv_folder, f"updated_{file}")
        df.to_csv(output_path, index=False)

        print(f"Saved: {output_path}")

print("\nALL FILES UPDATED SUCCESSFULLY!")