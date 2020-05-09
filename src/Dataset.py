import pandas as pd
import joblib
import numpy as np
from PIL import Image
import albumentations
import torch
import pickle
import os

class BengaliAiDataset:
    def __init__(self, folds, img_height, img_width, mean, std):
        df = pd.read_csv("../input/train_startified_fold5.csv")

        df = df[df.kfold.isin(folds)]
        self.image_ids = df.image_id.values
        self.grapheme_root = df.grapheme_root.values
        self.vowel_diacritic = df.vowel_diacritic.values
        self.consonant_diacritic = df.consonant_diacritic.values
        self.kfold = df.kfold.values

        self.img_height = img_height
        self.img_width = img_width

        # Just Validate 
        if len(folds) == 1:
            self.augmentations = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.Normalize(mean, std, always_apply = True)
        ])
        else:
            self.augmentations = albumentations.Compose([
                albumentations.Resize(img_height, img_width, always_apply=True),
                albumentations.Blur(blur_limit=(7, 7), p=0.5),
                albumentations.ShiftScaleRotate(shift_limit = 0.7, scale_limit = 0.5, rotate_limit = 10),
                albumentations.ElasticTransform(),
                albumentations.Normalize(mean, std, always_apply = True)        
                ])
    
    def __len__(self):
        return len(self.image_id)

    # Returns a augmented image 
    # TODO : Use Augmix here
    def image_load(self, item):
        item_array = joblib.load(f"../input/image_pickles/{self.image_ids[item]}.pkl")
        item_array = np.array(item_array[1:])
        # the image_size id 137X236
        item_array = item_array.reshape(self.img_height, self.img_width).astype(np.float32)
        image = Image.fromarray(item_array).convert("RGB")
        image = self.augmentations(image=np.array(image))["image"]
        # transpose the channels for torchvision model
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # Sanity Check
        assert image.shape == (3, self.img_height, self.img_width)
        return {
            "image" : torch.tensor(image, dtype = torch.float),
            "grapheme_root" : torch.tensor(self.grapheme_root[item], dtype = torch.long),
            "vowel_diacritic" : torch.tensor(self.vowel_diacritic[item], dtype = torch.long),
            "consonant_diacritic" : torch.tensor(self.consonant_diacritic[item], dtype = torch.long)
        }

    # return a PIL Object with the image_id item
    def __getitem__(self, item):
        return self.image_load(item)


if __name__ == "__main__":
    print(os.getcwd())
    # assert os.path.exists("../input/image_pickels/Train_65893.pkl")
    obj = BengaliAiDataset([0], 137, 236, 0.1, 1)
    d = obj[200]
    for k, v in d.items():
        print(k, v)
    