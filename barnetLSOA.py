from imd2SQL import dbPath
import sqlite3
import pandas as pd
from pathlib import Path


def barnetLsoaPath()-> Path:
    cwd = Path.cwd()
    dc2 = cwd.parent
    data = dc2.joinpath("data/barnet_lsoa_codes.csv")
    return data

def barnet_LSOA_series()->pd.Series:
    dropnaDBpath = dbPath()
    conn = sqlite3.connect(dropnaDBpath)
    lsoaCodesAndNamesQuery = """
    SELECT "LSOA code", "LSOA name"
    FROM table_name
    """
    lsoaCodesAndNamesDF = pd.read_sql(lsoaCodesAndNamesQuery, conn)
    lsoaCodesAndNamesDF.dropna(inplace=True)
    lsoaCodesAndNamesDF.drop_duplicates(inplace=True)
    print(len(lsoaCodesAndNamesDF))
    lsoaCodesAndNamesDF = lsoaCodesAndNamesDF[lsoaCodesAndNamesDF["LSOA name"].str.contains("Barnet")]
    print(len(lsoaCodesAndNamesDF))
    conn.close()
    return lsoaCodesAndNamesDF["LSOA code"]

def main():
    ser = barnet_LSOA_series()
    ser.to_csv(barnetLsoaPath())

if __name__ == "__main__":
    main()
    

    
