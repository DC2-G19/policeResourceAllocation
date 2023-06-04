import datetime
from tqdm import tqdm
import xgboost as xgb
from pathlib import Path
import sqlite3
import pandas as pd
from typing import Any, Tuple
import numpy as np



def dbPath() -> Path:
    cwd = Path.cwd()
    dc2 = cwd.parent
    db = dc2.joinpath("data/database_final.db")
    return db


def modelPath(lsoaCode:str) ->Path:
    cwd = Path.cwd()
    dc2 = cwd.parent
    modelPath = dc2.joinpath(f"data/models/{lsoaCode}.bin")
    return modelPath


def makeSunlightPrediction(date: datetime.datetime)-> Tuple[Any, Any]:

    conn = sqlite3.connect(dbPath())
    sunlightDF= pd.read_sql("SELECT * FROM sunlight", conn)
    sunlightDF["Year-Month"] = pd.to_datetime(sunlightDF["Year-Month"])
    month_dummies = pd.get_dummies(sunlightDF["Year-Month"].dt.month)
    sunlightDF = pd.concat([sunlightDF, month_dummies], axis=1)
    sunlightDF.drop(columns=["index"], inplace=True)
    for i in range(1,12):
        sunlightDF[f"Sunlight_shift_{i}"] = sunlightDF["Sunlight"].shift(i)

    sunlightDF = sunlightDF.set_index(sunlightDF["Year-Month"])
    sunlightDF.drop(columns = ["Year-Month"], inplace=True)

    sunlightDF.dropna(inplace=True)
    vals = sunlightDF[sunlightDF.index == date]
    # corr = vals["Sunlight"].values
    vals = vals.drop(columns=["Sunlight"])


    cwd = Path.cwd()
    dc2 = cwd.parent
    sunlightModelPath = dc2.joinpath("data/models/sunlight.bin")
    model = xgb.XGBRegressor()
    model.load_model(sunlightModelPath)
    pred = model.predict(vals.values)
    # return corr[0], pred[0]
    return pred[0]


def makeAllFeatures()-> Tuple[pd.DataFrame, list]:
    conn = sqlite3.connect(dbPath())
    query_lsoa_codes = "SELECT geogcode FROM lsoa_code_to_name"
    lsoa_codes = pd.read_sql(query_lsoa_codes, conn)

    query_unemployement = "SELECT * FROM montly_unemployement_claimant_count_by_lsoa_barnet"
    unemployement = pd.read_sql(query_unemployement, conn)

    unemployement.dropna(inplace=True)
    unemployement.drop(columns="index", inplace=True)
    unemployement = unemployement[(unemployement["date"]<"2020") & (unemployement["date"]>"2012")]

    unemployement = unemployement[unemployement["geogcode"]!="Column Total"].copy()
    lsoa_code_list = lsoa_codes[lsoa_codes["geogcode"] != "Column Total"]["geogcode"].to_list()
    allFeatures = pd.DataFrame(columns=lsoa_code_list, index=pd.to_datetime(unemployement["date"].unique()))

    for row in tqdm(unemployement.index):
        allFeatures[unemployement["geogcode"][row]][unemployement["date"][row]] = unemployement["value"][row]
    
    shiftColumnList = []
    for code in tqdm(lsoa_code_list):
        tempDF = pd.DataFrame(columns=[f"{code}_shift_{i+1}" for i in range(12)])
        for i in range(12):
            # allFeatures[f"{code}_shift_{i+1}"] = allFeatures[f"{code}"].shift(i+1)
            tempDF[f"{code}_shift_{i+1}"] = allFeatures[code].shift(i+1)
        shiftColumnList.append(tempDF.copy())
    allShifts = pd.concat(shiftColumnList, axis=1)
    # return allShifts, 0
    allFeatures = pd.concat([allFeatures, allShifts], axis=1)
    allFeatures = allFeatures.dropna()
    conn.close()
    return allFeatures, lsoa_code_list



def makeFirstUnemployementPrediction(allFeatures: pd.DataFrame,lsoa_code_list: list, date)-> pd.DataFrame:
    code_with_shift = allFeatures.drop(lsoa_code_list, axis=True).copy()
    vals = code_with_shift[code_with_shift.index == datetime.datetime.fromtimestamp(date)].copy()
    predDF = pd.DataFrame(index=[date], columns=lsoa_code_list)    
    for lsoaCode in tqdm(lsoa_code_list):
        model = xgb.XGBRegressor()
        model.load_model(modelPath(lsoaCode))
        predDF[lsoaCode][date] = model.predict(vals.values)

    return predDF

def makeUnemployementPrediction(x_arr: np.array,lsoa_code_list: list)-> np.array:
    predArray = np.empty(len(lsoa_code_list))
    for i, lsoaCode in tqdm(enumerate(lsoa_code_list)):
        model = xgb.XGBRegressor()
        model.load_model(modelPath(lsoaCode))
        predArray[i] = model.predict(x_arr)
    return predArray

def get_unemployement_predictions_for_range(start: datetime.datetime, end: datetime.datetime, allFeatures: pd.DataFrame, lsoa_code_list: list)-> pd.DataFrame:
    predDict = {}
    indices = list(iter(allFeatures.index))
    tail = allFeatures[allFeatures.index == indices[-1]]
    vals = tail.to_numpy()
    vals = vals[0]
    print(type(vals))
    first = True
    for month in pd.date_range(start, end, freq="M"):
        # print(type(month))
        if first:
            pred = makeFirstUnemployementPrediction(allFeatures, lsoa_code_list, month)
            pred_vals = pred.values
            predDict[month] = pred_vals
            print(type(vals))
            vals = np.concatenate(pred_vals, vals[0][:-len(lsoa_code_list)])
            first = False
        else:
            predArray = makeUnemployementPrediction(vals, lsoa_code_list)
            predDict[month] = predArray
            vals = np.concatenate(predArray, vals[:-len(lsoa_code_list)])
            first = False
    # print(np.shape(end))
    print("it worked")

def main():
    allFeatures, lsoa_code_list= makeAllFeatures()
    # print(allFeatures.head())
    # predRow = makeUnemployementPrediction(allFeatures, lsoa_code_list, datetime.datetime(2015,10,1))
    # print(predRow.head())
    get_unemployement_predictions_for_range(
            start=datetime.datetime(2015,10,1),
            end=datetime.datetime(2016,3,1),
            allFeatures=allFeatures,
            lsoa_code_list=lsoa_code_list
            )
if __name__ == "__main__":
    main()

#%%
