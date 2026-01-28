import pandas as pd
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

df = pd.read_csv('train.csv')
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'],
                                    random_state=42)  # Splits the training dataset into training and validation 80/20

source_folder = 'train_data'

for split_name, split_df in [('train', train_df), ('val', val_df)]:
    for _, row in split_df.iterrows():
        label = row['label']
        Path(f'dataset/{split_name}/{label}').mkdir(parents=True, exist_ok=True)

        src = row["file_name"]
        img_name = Path(row["file_name"]).name
        dst = f'dataset/{split_name}/{label}/{img_name}'
        shutil.copy(src,
                    dst)  # Finally the now 2 datasets are split physically according to the csv so they may work with yolo
        # I shall keep the original labels as folder names but for the sake of clarity 1 is AI generated and 0 is human generated
