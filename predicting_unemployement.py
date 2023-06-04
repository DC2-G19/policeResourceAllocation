from dateutil.relativedelta import relativedelta
import datetime
from tqdm import tqdm
import xgboost as xgb
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path


def dbPath()-> Path:
    cwd = Path.cwd()
    dc2 = cwd.parent
    db= dc2.joinpath("data/database_final.db")
    return db

def modelPath(lsoaCode:str) -> Path:
    cwd = Path.cwd()
    dc2 = cwd.parent
    modelPath = dc2.joinpath(f"data/models/{lsoaCode}.bin")
    return modelPath


def makeAllFeatures():
    conn = sqlite3.connect(dbPath())
    query_lsoa_codes = """
    SELECT geogcode
    FROM lsoa_code_to_name
    """
    lsoa_codes = pd.read_sql(query_lsoa_codes, conn)

    query_unemployement = """
    SELECT * 
    FROM montly_unemployement_claimant_count_by_lsoa_barnet
    """
    unemployement = pd.read_sql(query_unemployement, conn)

    unemployement.dropna(inplace=True)
    unemployement.drop(columns="index", inplace=True)
    unemployement = unemployement[(unemployement["date"]<"2020") & (unemployement["date"]>"2012")]

    unemployement = unemployement[unemployement["geogcode"]!="Column Total"].copy()
    lsoa_code_list = lsoa_codes[lsoa_codes["geogcode"] != "Column Total"]["geogcode"].to_list()
    allFeatures = pd.DataFrame(columns=lsoa_code_list, index=pd.to_datetime(unemployement["date"].unique()))

    for row in unemployement.index:
        allFeatures[unemployement["geogcode"][row]][unemployement["date"][row]] = unemployement["value"][row]
    
    shiftColumnList = []
    for code in lsoa_code_list:
        tempDF = pd.DataFrame(columns=[f"{code}_shift_{i+1}" for i in range(12)])
        for i in range(12):
            tempDF[f"{code}_shift_{i+1}"] = allFeatures[code].shift(i+1)
        shiftColumnList.append(tempDF.copy())
    allShifts = pd.concat(shiftColumnList, axis=1)
    allFeatures = pd.concat([allFeatures, allShifts], axis=1)
    allFeatures = allFeatures.dropna()
    conn.close()
    return allFeatures, lsoa_code_list


def main():
    print("LOADING UNEMPLOYEMENT")
    allFeatures, lsoa_code_list = makeAllFeatures()
    print("UNEMPLOYEMENT LOADED")
    lastRow = allFeatures.tail(1)
    present = pd.to_datetime(list(lastRow.index)[0])
    # print(present)
    lastArrArr = lastRow.to_numpy()
    lastArr = lastArrArr[0].T
    # names = lastRow.columns
    # print(lastArr[:].shape)
    pred = {"LSOA": lsoa_code_list}
    for i in tqdm(range(12)):
        tempGuess = np.empty(len(lsoa_code_list))
        for j, code in enumerate(lsoa_code_list):
            model = xgb.XGBRegressor()
            model.load_model(modelPath(code))
            X = lastArr[:-211].reshape(1,-1).copy()
            guess = model.predict(X)
            tempGuess[j] = guess[0]
        pred[pd.to_datetime(present+relativedelta(months=i+1))] = tempGuess
        last_arr_slice = lastArr[:-211].copy()
        lastArr = np.concatenate([tempGuess, last_arr_slice])


    return pd.DataFrame(pred)




if __name__ == "__main__":
    main()
