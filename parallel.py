import sqlite3
import pandas as pd
from tqdm import tqdm
from imd2SQL import dbPath
from similarCodes import crime_uk_per_month_dir, filter, filteredLSOApath, loadBarnetLSOA, get_ranking_IMD_by_LSOA
from pathlib import Path
from typing import Generator
import multiprocessing
import functools


def listFilesInDirectory(directory: Path) -> Generator:
    return directory.glob("*")


def process_directory(monthDir, un):
    df_list = []
    for region in listFilesInDirectory(monthDir):
        reg = pd.read_csv(region)
        df_list.append(reg)
    df_concat = pd.concat(df_list, ignore_index=True)
    filtered = df_concat[df_concat["LSOA code"].isin(un)]
    return filtered


def main():
    lsoa = loadBarnetLSOA()
    imd_by_lsoa_15 = get_ranking_IMD_by_LSOA(2015)
    filt15 = filter(imd_by_lsoa_15, lsoa)
    imd_by_lsoa_19 = get_ranking_IMD_by_LSOA(2019)
    filt19 = filter(imd_by_lsoa_19, lsoa)
    conced = pd.concat([filt15["LSOA Code"], filt19["LSOA Code"]], axis=1)  # Fix concatenation
    conced = conced.drop_duplicates()  # Drop duplicates
    conced = pd.concat([conced, lsoa])
    un = pd.DataFrame(conced.drop_duplicates().values, columns=["LSOA code"])
    un = un.set_index("LSOA code")  # Set "LSOA code" as the index
    un = pd.DataFrame(un.index.values, columns=["LSOA code"])  # Convert to dataframe
    conn = sqlite3.connect(dbPath())
    cursor = conn.cursor()

    # Drop the table if it exists
    cursor.execute("DROP TABLE IF EXISTS crimes_similar_LSOA")
    conn.commit()
    
    # multiprocessing to parallelize directory processing
    
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processes)
    process_directory_partial = functools.partial(process_directory, un=un)
    results = []
    for monthDir in tqdm(listFilesInDirectory(crime_uk_per_month_dir())):
        result = pool.apply_async(process_directory_partial, args=(monthDir,))
        results.append(result)
    pool.close()
    pool.join()

    finalDF = pd.concat([result.get() for result in results], ignore_index=True)
    finalDF.drop(["Context", "Last outcome category", "Crime ID"], axis=1, inplace=True)
    
    print(finalDF.info())
    finalDF.to_sql("crimes_similar_LSOA", conn, if_exists="append", index=False)
    conn.close()

if __name__ == "__main__":
    main()
