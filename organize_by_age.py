import os
import pandas as pd
from sklearn.model_selection import train_test_split

preprocessed_dir = 'preprocessed'
sorted_dir = 'IXI-T1-sorted'

# Build a mapping from image filename to class (age group)
filename_to_class = {}
for class_name in os.listdir(sorted_dir):
    class_path = os.path.join(sorted_dir, class_name)
    if not os.path.isdir(class_path):
        continue
    for fname in os.listdir(class_path):
        if fname.endswith('.nii.gz'):
            filename_to_class[fname] = class_name

rows = []
for fname in os.listdir(preprocessed_dir):
    if fname.endswith('_preprocessed.nii.gz'):
        # Recover original filename (before preprocessing)
        orig_fname = fname.replace('_preprocessed', '')
        class_name = filename_to_class.get(orig_fname, 'Unknown')
        rows.append({
            'ImagePath': os.path.join(preprocessed_dir, fname),
            'ImageId': os.path.splitext(os.path.splitext(fname)[0])[0],
            'Class': class_name
        })

df = pd.DataFrame(rows)

# Remove classes with fewer than 3 images
class_counts = df['Class'].value_counts()
valid_classes = class_counts[class_counts >= 3].index
removed_classes = class_counts[class_counts < 3].index.tolist()
if removed_classes:
    print(f"Removing classes with <3 images: {removed_classes}")
df = df[df['Class'].isin(valid_classes)]

df.to_csv('preprocessed_images.csv', index=False)
print('CSV created: preprocessed_images.csv')

# Try stratified split, fallback to random if it fails
try:
    train_val, test = train_test_split(df, test_size=0.3, stratify=df['Class'], random_state=42)
    train, val = train_test_split(train_val, test_size=1/6, stratify=train_val['Class'], random_state=42)
except ValueError as e:
    print(f"Stratified split failed: {e}")
    print("Falling back to random split (no stratification).")
    train_val, test = train_test_split(df, test_size=0.3, random_state=42)
    train, val = train_test_split(train_val, test_size=1/6, random_state=42)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)
val.to_csv('val.csv', index=False)
print(f"Train: {len(train)}, Test: {len(test)}, Val: {len(val)}") 