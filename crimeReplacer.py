import sqlite3
import pandas as pd
from pathlib import Path
from imd2SQL import dbPath

def getCleanPath()-> Path:
    cwd = Path.cwd()
    parent = cwd.parent
    cleanPath = parent.joinpath("data/unzipped/conc_clean.csv")
    return cleanPath

def finalDbPath():
    cwd = Path.cwd()
    parent = cwd.parent
    cleanPath = parent.joinpath("data/database_final.db")
    return cleanPath
    

def main():
    # conn = sqlite3.connect(finalDbPath())
    conn = sqlite3.connect(dbPath())

    print("CONNECTED")
    cleanCrime = pd.read_csv(getCleanPath())
    print("READ")
    cleanCrime.to_sql("table_name", conn, if_exists="replace", index=False)
    print("LOADED")
    conn.close()
    print("CONNECTION CLOSED")

if __name__ == "__main__":
    main()
