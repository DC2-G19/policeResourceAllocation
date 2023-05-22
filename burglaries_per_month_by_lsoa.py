import sqlite3 
import pandas as pd 
from tqdm import tqdm 
import matplotlib.pyplot as plt
from pathlib import Path

def get_final_db_path() -> Path:
    cwd = Path.cwd()
    parent = cwd.parent
    db_final_path = parent.joinpath("data/database_final.db")
    return db_final_path

def pickPath()-> Path:
    cwd = Path.cwd()
    parent = cwd.parent
    picklePath = parent.joinpath("data/pickle.pkl")
    return picklePath


def main():
    conn = sqlite3.connect(get_final_db_path())
    query_codes_and_names = """
    SELECT geogcode
    FROM lsoa_code_to_name
    """
    lsoa_codes_names = pd.read_sql(query_codes_and_names, conn)
    lsoa_codes_names_list = lsoa_codes_names["geogcode"].to_list()
    joined = "', '".join(lsoa for lsoa in lsoa_codes_names_list[:])
    query = f"""
    SELECT Month, [LSOA code]
    FROM table_name
    WHERE [LSOA code] IN ('{joined}') AND [Crime type] = 'Burglary'
    """
    burglaries = pd.read_sql(query, conn)
    # print(burglaries.info())
    # print(burglaries.head())
    burglaries["Month"] = pd.to_datetime(burglaries["Month"])
    burglaries_month_by_lsoa = {}
    for code in tqdm(burglaries["LSOA code"].unique()):
        frame = burglaries[burglaries["LSOA code"] == code]["Month"].copy()
        # print(frame.value_counts())
        burglaries_month_by_lsoa[code] = frame.value_counts()

    pd.to_pickle(burglaries_month_by_lsoa, pickPath())
    print("PICKLING IS DONE")
    conn.close()


if __name__ == "__main__":
    main()
