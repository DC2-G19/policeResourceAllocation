from pathlib import Path
import pandas as pd
import sqlite3
from typing import List


def get_barnet_data()-> List:
    cwd = Path.cwd()
    parent = cwd.parent
    barnetDir = parent.joinpath(
            "data/unzipped/unemployement/barnetTest"
            )
    return [file for file in barnetDir.iterdir()]

def dbPath()-> Path:
    cwd = Path.cwd()
    parent = cwd.parent
    dbPath = parent.joinpath("data/database_final.db")
    return dbPath

def main():
    """
    puts the monthly unemployement data from all the LSOAs in barnett into the SQL database.
    """
    conn = sqlite3.connect(dbPath())
    dirContent = get_barnet_data()
    
    data = dirContent[1]
    # print(data)
    unemployementDF = pd.read_csv(data, delimiter="\t")
    del unemployementDF["Age"]
    del unemployementDF["Gender"]
    del unemployementDF["measure"]
    del unemployementDF["flag"]
    del unemployementDF["value type"]
    unemployementDF["date"] = pd.to_datetime(unemployementDF["date"])
    unemployementDF.to_sql("montly_unemployement_claimant_count_by_lsoa_barnet", conn, if_exists="replace")

    geog = dirContent[0]
    geogDF = pd.read_csv(geog, delimiter="\t", index_col="geogcode")
    geogDF.to_sql("lsoa_code_to_name", conn, if_exists="replace")

    conn.close()

if __name__ == "__main__":
    main()

