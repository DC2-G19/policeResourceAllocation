from imd2SQL import dbPath
import sqlite3
import pandas as pd


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
    conn.close()
    return lsoaCodesAndNamesDF["LSOA code"]

def main():
    ser = barnet_LSOA_series()
    ser.to_csv(dbPath().joinpath("barnet_lsoa_codes.csv"))

if __name__ == "__main__":
    main()
    

    
