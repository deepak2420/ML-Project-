import pandas as pd
import os

def load_dataset(data_path):

    csv_files = [
        f for f in os.listdir(data_path)
        if f.endswith(".csv") and not f.startswith(".")
    ]

    if len(csv_files) == 0:
        raise Exception("No CSV files found in dataset folder")

    dataframes = []

    for file in csv_files:
        file_path = os.path.join(data_path, file)
        print("Loading:", file)

        try:
            df = pd.read_csv(file_path)
            dataframes.append(df)

        except Exception as e:
            print(f"Error loading {file}: {e}")

    data = pd.concat(dataframes, ignore_index=True)

    print("Dataset Loaded Successfully")
    print("Dataset Shape:", data.shape)

    return data