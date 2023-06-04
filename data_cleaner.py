#Imports
import os
import shutil
import pandas as pd
import sqlite3
from pathlib import Path
from pathFunc import dbPath, dataDir


def data_to_df(file):
    #Filters out the required excel files (makes a copy). Make sure you are in the correct dir where your data is stored
    dataset_dir = dataDir().joinpath(file)  # replace with the path to your dataset folder
    filtered_dir = str(dataset_dir) + "CLEANED"  # replace with the path to the folder where you want to store the filtered data
    if not os.path.exists(filtered_dir):
        os.makedirs(filtered_dir)

    for month_folder in os.listdir(dataset_dir):
        if not month_folder.startswith("20"):  # skip any folders that don't start with a date
            continue
        for csv_file in os.listdir(os.path.join(dataset_dir, month_folder)):
            if not csv_file.endswith("-metropolitan-street.csv"):  # skip any files that don't match the desired format
                continue
            source_path = os.path.join(dataset_dir, month_folder, csv_file)
            destination_path = os.path.join(filtered_dir, csv_file)
            shutil.copy(source_path, destination_path)  # copy the matching file to the filtered_data folder


    folder_path = filtered_dir  # replace with the path to your Excel files folder
    df_list = []
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):  # only read in CSV files
            file_path = os.path.join(folder_path, file)
            df = pd.read_csv(file_path)
            df_list.append(df)
    combined_df_temp = pd.concat(df_list, ignore_index=True)
    combined_df = combined_df_temp.drop(['Context', 'Last outcome category', 'Crime ID' ], axis=1)
    return combined_df

def df_to_SQLdb(path, df):
    # Connect to an SQLite database (creates a new database file if it doesn't exist)
    conn = sqlite3.connect(path)

    # Write DataFrame to an SQL database table
    df.to_sql('table_name', conn, if_exists='replace', index=False)

    # Close the database connection
    conn.close()

def df_to_csv(path, df):
    df.to_csv(path, index=False)  # save the combined DataFrame to a new CSV file

def raw_data_to_file(save_path, data_relative_path):
    df=data_to_df(data_relative_path)
    df_to_csv(save_path, df)
    
raw_data_to_file('cleaned.csv', '2023-04')
