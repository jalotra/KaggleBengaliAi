# Use this file to create folds in the train data set 
import os
import pandas as pd
import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

if __name__ == "__main__":
    # Load the train dataset
    df = pd.read_csv("../input/train.csv")

    df.loc[:, "kfold"] = -1
    # Shuffle the Rows
    df = df.sample(frac = 1).reset_index(drop = True)

    print(df.head())

    # Stratified K-fold
    X = df.image_id.values
    Y = df[["grapheme_root", "vowel_diacritic", "consonant_diacritic"]].values
    
    mskf = MultilabelStratifiedKFold(n_splits= 5, random_state = None)
    for fold, (train_idx, validation_idx) in enumerate(mskf.split(X, Y)):
        print("TRAIN : ", train_idx, "VALIDATION : ", validation_idx)
        # Tag the rows of the validation indexes with fold id else -1 for train_idx
        df.loc[validation_idx, "kfold"] = fold
    
    print(df.kfold.value_counts())

    df.to_csv("../input/train_startified_fold5.csv", index = False)
