import sqlite3
import pandas as pd
from imd2SQL import dbPath
from pathlib import Path

def sunlight_path()-> Path:
    cwd = Path.cwd()
    parent = cwd.parent
    sunlightPath = parent.joinpath("data/unzipped/sunlight_updated.csv")
    return sunlightPath


def main():
    conn = sqlite3.connect(dbPath()) 
    sunlightDF = pd.read_csv(sunlight_path())
    sunlightDF["Year-Month"] = pd.to_datetime(sunlightDF["Year-Month"])
    sunlightDF.to_sql("sunlight", conn, if_exists="replace")
    conn.close()


if __name__ == "__main__":
    main()
