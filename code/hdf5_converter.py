# Description: This script reads the CSV file containing image labels and the image directory, loads the images,
#              resizes them, and saves them to an HDF5 file.

import os
import h5py
import pandas as pd
import numpy as np
from PIL import Image

# Paths to your CSV file and image directory
csv_file = '../data/train_labels.csv'
root_dir = '../data/fundus_757/'

# Desired image size
image_size = (224, 224)

# Read CSV file
labels = pd.read_csv(csv_file)
data = []
label_list = []

# Load images and labels
for idx in range(len(labels)):
    img_name = os.path.join(root_dir, labels.iloc[idx, 0])
    if os.path.isfile(img_name):
        image = Image.open(img_name).convert('RGB')
        image = image.resize(image_size)  # Resize image
        image = np.array(image)
        data.append(image)
        label_list.append(labels.iloc[idx, 1])

data = np.stack(data)  # Stack images into a single NumPy array
label_list = np.array(label_list)

# Save to HDF5 file
with h5py.File('dataset.h5', 'w') as hf:
    hf.create_dataset('images', data=data)
    hf.create_dataset('labels', data=label_list)

print("Conversion to HDF5 complete. File saved as 'dataset.h5'")
