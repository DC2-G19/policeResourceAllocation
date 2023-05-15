import sqlite3
import pandas as pd
from pandas.core.series import Axis
from tqdm import tqdm
from imd2SQL import dbPath
from similarCodes import crime_uk_per_month_dir, filter, filteredLSOApath, loadBarnetLSOA, get_ranking_IMD_by_LSOA
from pathlib import Path
from typing import Generator


def df_to_SQLdb(df):
    # Connect to an SQLite database (creates a new database file if it doesn't exist)
    conn = sqlite3.connect(dbPath())

    # Write DataFrame to an SQL database table
    df.to_sql('crimes_similar_LSOA', conn, if_exists='append', index=False)

    # Close the database connection
    conn.close()

def listFilesInDirectory(directory: Path) -> Generator:
    return directory.glob("*")



def main():
    lsoa = loadBarnetLSOA()
    imd_by_lsoa_15 = get_ranking_IMD_by_LSOA(2015)
    filt15 = filter(imd_by_lsoa_15, lsoa)
    imd_by_lsoa_19 = get_ranking_IMD_by_LSOA(2019)
    filt19 = filter(imd_by_lsoa_19, lsoa)
    conced = pd.concat([filt15["LSOA Code"], filt19["LSOA Code"]])
    conced = pd.concat([conced, lsoa])
    un = conced.drop_duplicates()
    
    conn = sqlite3.connect(dbPath())

    for monthDir in tqdm(listFilesInDirectory(crime_uk_per_month_dir())):
        df = []
        for region in listFilesInDirectory(monthDir):
            reg = pd.read_csv(region)
            df.append(reg)

        conced = pd.concat(df, ignore_index=True)
        filtered = conced[conced["LSOA code"].isin(un)]
        filtered.drop(["Context", "Last outcome category", "Crime ID"], axis=1, inplace=True)
        # print(filtered.info() )
        filtered.to_sql('crimes_similar_LSOA', conn, if_exists='append', index=False)
         

    conn.close()

if __name__ == "__main__":
    main()
