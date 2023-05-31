import pandas as pd
import sqlite3
from pathlib import Path
from unemployement_to_sql import dbPath


def getHousing_combined()->pd.DataFrame:
    cwd = Path.cwd()
    parent = cwd.parent
    housing_path = parent.joinpath("data/unzipped/housing_combined.xlsx")
    return pd.read_excel(housing_path)



def main():
    housing = getHousing_combined()
    # print(housing.columns)
    conn = sqlite3.connect(dbPath())
    housing.to_sql("housing_by_lsoa", conn, if_exists="replace")
    conn.close()

if __name__ == "__main__":
    main()
