from pathlib import Path
from typing import Set, Generator
from barnetLSOA import barnetLsoaPath
import pandas as pd
import sqlite3
from imd2SQL import dbPath
import matplotlib.pyplot as plt


def crime_uk_per_month_dir() -> Path:
    cwd = Path.cwd()
    dc2 = cwd.parent
    crimeCSV = dc2.joinpath("data/unzipped/crimeCSV")
    return crimeCSV


def listFilesInDirectory(directory: Path) -> Generator:
    return directory.glob("*")


def loadBarnetLSOA():
    col = pd.read_csv(barnetLsoaPath())["LSOA code"]
    return col


def get_ranking_IMD_by_LSOA(year: int):
    db = sqlite3.connect(dbPath())
    if year == 2015:
        query = """
        SELECT *
        FROM IOD_feature_code_rank_15
        """
    elif year == 2019:
        query = """
        SELECT *
        FROM IOD_feature_code_rank_19
        """
    else:
        print("INPUT YEAR INVALID")
        return 0
    df = pd.read_sql(query, db)
    return df #.set_index("LSOA Code")

def filter(imd_by_lsoa: pd.DataFrame, lsoa: pd.Series) -> pd.DataFrame:
    barnetIMD = imd_by_lsoa[imd_by_lsoa["LSOA Code"].isin(lsoa)]
    mean = barnetIMD["f. Crime Domain"].mean()
    std = barnetIMD["f. Crime Domain"].std() 
    filt_range = (round(mean - std), round(mean + std))
    filt = imd_by_lsoa[imd_by_lsoa["f. Crime Domain"].isin(range(filt_range[0],filt_range[1]))]
    return filt

def filteredLSOApath()-> Path:
    cwd = Path.cwd()
    dc2 = cwd.parent
    return dc2.joinpath("data/selectLSOA.csv")


def main():
    lsoa = loadBarnetLSOA()
    imd_by_lsoa_15 = get_ranking_IMD_by_LSOA(2015)
    filt15 = filter(imd_by_lsoa_15, lsoa)
    imd_by_lsoa_19 = get_ranking_IMD_by_LSOA(2019)
    filt19 = filter(imd_by_lsoa_19, lsoa)
    conced = pd.concat([filt15["LSOA Code"], filt19["LSOA Code"]])
    conced = pd.concat([conced, lsoa])
    un = conced.drop_duplicates()
    un.reset_index(inplace=True, drop=True)
    un.to_csv(filteredLSOApath(), index=False)
    

if __name__ == "__main__":
    main()
