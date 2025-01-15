import os
import shutil
import random

input_folder = r"C:\Users\Ehtisham\OneDrive\Desktop\AI\AI\dataset\images\train"
output_folder = "dataset_split"
train_folder = os.path.join(output_folder, "train")
val_folder = os.path.join(output_folder, "val")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

all_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f))]
random.seed(42)
random.shuffle(all_files)

split_idx = int(0.8 * len(all_files))
train_files = all_files[:split_idx]
val_files = all_files[split_idx:]

for file in train_files:
    shutil.copy(os.path.join(input_folder, file), train_folder)

for file in val_files:
    shutil.copy(os.path.join(input_folder, file), val_folder)

print("Dataset split into train and val!")
