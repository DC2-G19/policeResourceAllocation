import pathlib
import pandas as pd
from pathlib import Path
import sqlite3
import datetime as dt
from typing import Tuple


def getUnemployementDir() -> Tuple[pathlib.PosixPath, pathlib.PosixPath]:
    cwd = Path.cwd()
    parent = cwd.parent
    unemployement_csv_dir= parent.joinpath("data/unzipped/unemployement")
    ls = [file for file in unemployement_csv_dir.iterdir()]
    
    return ls[0], ls[1] 

def dbPath() -> Path:
    cwd = Path.cwd()
    parent = cwd.parent
    dbPath = parent.joinpath("data/database_conc_na.db")
    return dbPath

def displayNA(dataDF: pd.DataFrame)-> None:
    """
    prints amount of NA per datasets
    """
    for sex in dataDF["Gender"].unique():
        for measure in dataDF["measure"].unique():
            temp = dataDF[(dataDF["Gender"] == sex) & (dataDF["measure"] == measure)]
            print(f"s = {sex}, m = {measure}, na = {temp['value'].isna().sum()}")


def  splitter(df, crit:str) -> pd.DataFrame:
    temp = df[(df["Gender"] == crit) & (df["measure"]== "Claimant count")].copy()

    return temp





def main():
    dataPath, geoPath = getUnemployementDir()


    dataDF = pd.read_csv(dataPath, delimiter="\t")
    geoDF = pd.read_csv(geoPath, delimiter="\t",index_col="geogcode")


    del dataDF["Age"]
    del dataDF["value type"]
    del dataDF["flag"]
    dataDF["date"] = pd.to_datetime(dataDF['date'])
    
    dbpth = dbPath().__str__()
    conn = sqlite3.connect(dbpth)
    geoDF.to_sql("ward_code_to_name", conn, if_exists="replace")    
    for sex in dataDF["Gender"].unique():
        temp =  splitter(dataDF, crit="Total")
        temp.to_sql(f"unemployement_{sex}", conn, if_exists="replace", index=False)
    
    conn.close()
    
if __name__ == "__main__":
    main()
