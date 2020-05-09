# This file is to factor the parquet files
import os
import sys
import pandas as pd
from tqdm import tqdm
import joblib
import glob
from joblib import Parallel
import pickle

def create_pickles(filepath):
    # train_parquets = glob.glob("../input/train_*.parquet")
    
    print(filepath)
    df = pd.read_parquet(filepath)
    image_ids = df.image_id.values
    df.drop("image_id", axis = 1)

    # convert the df to arr
    image_arr = df.values
    for j, image_id in tqdm(enumerate(image_ids), total = len(image_ids)):
        joblib.dump(image_arr[j, :], f"../input/image_pickles/{image_id}.pkl")

if __name__ == "__main__":
    for file in glob.glob("../input/train_*.parquet"):
        create_pickles(file)