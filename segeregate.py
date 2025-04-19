import os
import shutil
import pandas as pd

# Paths
image_dir = r'C:\Users\Pankaj\Desktop\final_copy_skin\ham10000\HAM10000_images_part_2'
csv_path = r'C:\Users\Pankaj\Desktop\final_copy_skin\ham10000\HAM10000_metadata.csv'
output_dir = 'segregated_images'

# Load CSV
df = pd.read_csv(csv_path)

# Map labels to full names
label_map = {
    'mel': 'Melanoma',
    'nv': 'Nevus',
    'bkl': 'Benign Keratosis-like Lesions',
    'bcc': 'Basal Cell Carcinoma',
    'akiec': 'Actinic Keratoses',
    'vasc': 'Vascular Lesions',
    'df': 'Dermatofibroma'
}

# Create folders
for label in label_map.values():
    os.makedirs(os.path.join(output_dir, label), exist_ok=True)

# Move images
for _, row in df.iterrows():
    img_name = row['image_id'] + '.jpg'
    label = label_map[row['dx']]
    src = os.path.join(image_dir, img_name)
    dst = os.path.join(output_dir, label, img_name)
    if os.path.exists(src):
        shutil.copy(src, dst)
