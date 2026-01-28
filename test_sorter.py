# Very close to the splitter code I used for train and val
# Since my dataset is part of a kaggle competition I do not have public ground truths nor is the test folder "sorted"
# I shall be using the output that scored 96.65% accuracy as I cannot annotate the 5k pictures of the test dataset by hand
# This is the best scored solution for the available test dataset as it was seemingly changed mid-competition

import pandas as pd
import shutil
from pathlib import Path

df = pd.read_csv('test.csv')

source_folder = 'test_data_v2'  # Once again to avoid duplicates cramming my diskspace the original test data has been removed

for _, row in df.iterrows():
    label = row['label']
    Path(f'dataset/test/{label}').mkdir(parents=True, exist_ok=True)

    src = row["id"]
    img_name = Path(row["id"]).name
    dst = f'dataset/test/{label}/{img_name}'
    shutil.copy(src, dst)
