import os
import ast
from model_dispatcher import MODEL_DISPATCHER
from Dataset import BengaliAiDataset
import torch
from tqdm import tqdm
from earlystopping import EarlyStopping

DEVICE = os.environ.get("DEVICE")
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

def loss_fn(outputs, targets):
    loss = 0
    for i in range(3):
        loss += torch.nn.CrossEntropyLoss()(outputs[i], targets[i])

    return loss/3

def train(dataset, dataLoader, model, optimiser):
    model.train()

    for batch, return_dict in tqdm(enumerate(dataLoader), total = len(dataset)/dataLoader.batch_size):
        image = return_dict["image"]
        grapheme_root = return_dict["grapheme_root"]
        vowel_diacritic = return_dict["vowel_diacritic"]
        consonant_diacritic = return_dict["consonant_diacritic"]

        image = image.to(DEVICE, dtype = torch.float)
        grapheme_root = grapheme_root.to(DEVICE, dtype = torch.long)
        vowel_diacritic = vowel_diacritic.to(DEVICE, dtype = torch.long)
        consonant_diacritic = consonant_diacritic.to(DEVICE, dtype = torch.long)

        optimiser.zero_grad()
        outputs = model(image)
        targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
        loss = loss_fn(outputs, targets)

        loss.backward()
        optimiser.step()

def evaluate(dataset, dataloader, model, optimiser):
    model.eval()

    final_loss = 0
    counter = 0
    with torch.no_grad():
        for batch, return_dict in tqdm(enumerate(dataloader), total = len(dataset)/dataloader.batch_size):
            image = return_dict["image"]
            grapheme_root = return_dict["grapheme_root"]
            vowel_diacritic = return_dict["vowel_diacritic"]
            consonant_diacritic = return_dict["consonant_diacritic"]

            image = image.to(DEVICE, dtype = torch.float)
            grapheme_root = grapheme_root.to(DEVICE, dtype = torch.long)
            vowel_diacritic = vowel_diacritic.to(DEVICE, dtype = torch.long)
            consonant_diacritic = consonant_diacritic.to(DEVICE, dtype = torch.long)

            outputs = model(image)
            targets = (grapheme_root, vowel_diacritic, consonant_diacritic)
            loss = loss_fn(outputs, targets)
            final_loss += loss
            counter += 1
    
    # Get the mean loss
    return final_loss/counter

def main():
    is_training = True
    model = MODEL_DISPATCHER[BASE_MODEL](is_training)
    model.to(DEVICE)
    EarlyStoppingObject = EarlyStopping()

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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode = "min", \
                                                            patience = 5, factor = 0.3, \
                                                                verbose = True)


    for epoch in enumerate(EPOCHS):
        train(Training_Dataset, Train_DataLoader, model, optimiser)
        validationScore = evaluate(Validation_Dataset, Validation_DataLoader, model, optimiser)
        scheduler.step(validationScore)
        print(f"EPOCH : {epoch} VALIDATION SCORE : {validationScore}")
        # torch.save(model.state_dict(), f"../input/output_models/{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin")
        EarlyStoppingObject(validationScore, model, f"../input/output_models/{BASE_MODEL}_fold{VALIDATION_FOLDS[0]}.bin")


if __name__ == "__main__":
    main()





    
    

