import pandas as pd
import sqlite3
from IMD2SQL import dbPath
import matplotlib.pyplot as plt
from pathlib import Path

def imagePath():
    cwd = Path.cwd()
    parent = cwd.parent
    imgDir = parent.joinpath("data/img")
    return imgDir

def burglaryDF() -> pd.DataFrame:
    conn = sqlite3.connect(dbPath())
    burglary_query = """SELECT *
    FROM table_name
    WHERE [Crime type] = "Burglary"
    """
    df = pd.read_sql_query(burglary_query, conn)
    conn.close()
    return df

def imd2015DF() -> pd.DataFrame:
    conn = sqlite3.connect(dbPath())
    get2015 = """
    SELECT *
    FROM IOD_feature_code_score_15
    """
    df = pd.read_sql_query(get2015, conn)
    conn.close()
    return df

def imd2019DF() -> pd.DataFrame:
    conn = sqlite3.connect(dbPath())
    get2019 = """
    SELECT *
    FROM IOD_feature_code_score_19
    """
    df = pd.read_sql_query(get2019, conn)
    conn.close()
    return df


def main() -> None:
    burglary = burglaryDF()
    iod15 = imd2015DF()
    iod19 = imd2019DF()
    iod15.set_index(["LSOA Code"], inplace=True)
    iod19.set_index(["LSOA Code"], inplace=True)
    crime_count = burglary.groupby("LSOA code").count()["Month"]
    merged15 = pd.merge(iod15, crime_count, left_index=True, right_index=True)
    merged19 = pd.merge(iod19, crime_count, left_index=True, right_index=True)
    fig, ax = plt.subplots(2,10, sharey=True, figsize= (60,20))
    for i, col in enumerate(iod15.columns):
        ax[0, i].scatter(merged15[col], merged15["Month"]);
        ax[1, i].scatter(merged19[col], merged19["Month"]);
    plt.savefig(imagePath().joinpath("LSOAplot.png"))


if __name__ == "__main__":
    main()

