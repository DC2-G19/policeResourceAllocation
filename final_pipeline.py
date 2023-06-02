import datetime
from tqdm import tqdm
import xgboost as xgb
from pathlib import Path
import sqlite3
import pandas as pd
from typing import Any


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


def makeSunlightPrediction(date: datetime.datetime)-> tuple[Any, Any]:

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

def makeUnemployementPrediction(date: datetime.datetime, lsoaCode :str)-> float:
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

    for code in tqdm(lsoa_code_list):
        for i in range(12):
            allFeatures[f"{code}_shift_{i+1}"] = allFeatures[f"{code}"].shift(i+1)

    allFeatures = allFeatures.dropna()
    # code_out_shift = allFeatures[lsoa_code_list].copy()
    code_with_shift = allFeatures.drop(lsoa_code_list, axis=True).copy()
    # corr = code_out_shift[code_out_shift.index == date][lsoaCode]
    vals = code_with_shift[code_with_shift.index == date]
    model = xgb.XGBRegressor()
    model.load_model(modelPath(lsoaCode))
    prediction = model.predict(vals.values)
    # return corr.values[0], prediction[0]
    return prediction[0]



