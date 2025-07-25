import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import nibabel as nib
import numpy as np

class MRIDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        # Map class names to integer labels
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(self.data['Class'].unique()))}
        self.data['Label'] = self.data['Class'].map(self.class_to_idx)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = row['ImagePath']
        img = nib.load(img_path).get_fdata()
        # Normalize to zero mean, unit variance
        img = (img - np.mean(img)) / (np.std(img) + 1e-8)
        # Add channel dimension: [1, D, H, W]
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img, dtype=torch.float32)
        label = int(row['Label'])
        if self.transform:
            img = self.transform(img)
        return img, label

def get_dataloader(csv_file, batch_size=2, shuffle=True, num_workers=0, transform=None):
    dataset = MRIDataset(csv_file, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader, dataset.class_to_idx

if __name__ == '__main__':
    # Example usage
    train_loader, class_to_idx = get_dataloader('train.csv', batch_size=2, shuffle=True)
    for imgs, labels in train_loader:
        print('Batch images shape:', imgs.shape)   # [B, 1, D, H, W]
        print('Batch labels:', labels)
        break
    print('Class to index mapping:', class_to_idx) 