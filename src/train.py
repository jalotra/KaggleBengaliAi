import os
import ast
from model_dispatcher import MODEL_DISPATCHER
from Dataset import BengaliAiDataset
import torch

DEVICE = "cpu"
TRAINING_FOLDS_CSV = os.environ.get("TRAINING_FOLDS_CSV")
IMG_HEIGHT = int(os.environ.get("IMG_HEIGHT"))
IMG_WIDTH = int(os.environ.get("IMG_WIDTH")) 
EPOCHS = int(os.environ.get("EPOCHS")) 

TRAIN_BATCH_SIZE = int(os.environ.get("TRAIN_BATCH_SIZE")) 
TEST_BATCH_SIZE = int(os.environ.get("TEST_BATCH_SIZE")) 

MODEL_MEAN = int(os.environ.get("MODEL_MEAN")) 
MODEL_STD = int(os.environ.get("MODEL_STD")) 

TRAINING_FOLDS = ast.literal_eval(os.environ.get("TRAINING_FOLDS"))
VALIDATION_FOLDS = ast.literal_eval(os.environ.get("VALIDATION_FOLDS"))
BASE_MODEL = os.environ.get("BASE_MODEL")

##########################################################################################
# TRAIN_LOOP STARTS FROM HERE

def train():
    is_training = True
    model = MODEL_DISPATCHER[BASE_MODEL](is_training)
    model.to(DEVICE)

    Training_Dataset = BengaliAiDataset(
        folds = TRAINING_FOLDS, \
        img_height= IMG_HEIGHT, \
        img_width= IMG_WIDTH, \
        mean = MODEL_MEAN,\
        std = MODEL_STD)

    Train_DataLoader = torch.utils.data.DataLoader(
        dataset = Training_Dataset,
        batch_size = TRAIN_BATCH_SIZE,
        shuffle = True,
        num_workers = 4 
    )

    Validation_Dataset = BengaliAiDataset(
        folds = VALIDATION_FOLDS, \
        img_height= IMG_HEIGHT, \
        img_width= IMG_WIDTH, \
        mean = MODEL_MEAN,\
        std = MODEL_STD)

    Validation_DataLoader = torch.utils.data.DataLoader(
        dataset = Validation_Dataset,
        batch_size = TEST_BATCH_SIZE,
        shuffle = False,
        num_workers = 4 
    )


    optimiser = torch.optim.Adam(model.parameters(), lr = 1e-4)
    scheduler = torch.optim.lr




    
    

