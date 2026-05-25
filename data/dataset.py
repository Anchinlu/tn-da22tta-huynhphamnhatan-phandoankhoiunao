import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

class BrainTumorDataset(Dataset):
    def __init__(self, dataset_dir, image_size=256, transform=None, patient_list=None):
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.transform = transform
        
        self.image_paths = []
        self.mask_paths = []
        
        if patient_list is not None:
            patients = patient_list
        else:
            patients = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
            
        for patient_dir in patients:
            patient_path = os.path.join(dataset_dir, patient_dir)
            
            if os.path.isdir(patient_path):
                masks = glob.glob(os.path.join(patient_path, '*_mask.tif'))
                
                for mask_path in masks:
                    img_path = mask_path.replace('_mask.tif', '.tif')
                    
                    if os.path.exists(img_path):
                        self.image_paths.append(img_path)
                        self.mask_paths.append(mask_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        mask_path = self.mask_paths[index]
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        image = image.resize((self.image_size, self.image_size), resample=Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), resample=Image.NEAREST)
        
        image = np.array(image)
        mask = np.array(mask)
        
        mask = (mask > 0).astype(np.float32)
        
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
            
        image = TF.to_tensor(image) 
        mask = torch.from_numpy(mask).unsqueeze(0) 
        
        return image, mask
