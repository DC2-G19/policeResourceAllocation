
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import sqlite3
from typing import Tuple, Dict, List
#%%
def dbPath()-> Path:
    cwd = Path.cwd()
    parent = cwd.parent
    return parent.joinpath("data/database_conc_na.db")

def getIMD() -> Tuple[pd.DataFrame, pd.DataFrame]:
    cwd = Path.cwd()
    parent = cwd.parent
    DLUHC_DIR= parent.joinpath("data/unzipped/DLUHC_open_data")
    imd2015path = DLUHC_DIR.joinpath('imd2015lsoa.csv')
    imd2019path = DLUHC_DIR.joinpath('imd2019lsoa.csv')
    imd2015DF = pd.read_csv(imd2015path)
    imd2019DF = pd.read_csv(imd2019path)
    return imd2015DF, imd2019DF
imd2015DF, imd2019DF = getIMD()

def remove_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["Units", "DateCode"])

def threeWaySplit(df:pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scoreDF = df[df["Measurement"] == "Score"]
    rankDF = df[df["Measurement"] == "Rank"]
    decileDF = df[df["Measurement"] == "Decile"]
    return scoreDF, rankDF, decileDF

def makePreprocessedDf(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["Indices of Deprivation", "FeatureCode"])["Value"].first().unstack(fill_value=0)

def makeRow(preprocessed_df: pd.DataFrame, id: str, keys: List) -> Dict:
    return {index: preprocessed_df.loc[index, id] for index in keys}

def inputRow(target: pd.DataFrame, origin: pd.DataFrame) -> pd.DataFrame:
    preprocessed_df = makePreprocessedDf(origin)
    keys = preprocessed_df.index.values.tolist()
    for id in tqdm(origin["FeatureCode"].unique()):
        row = makeRow(preprocessed_df, id, keys)
        target.loc[id] = row
    return target

def preProcessing(origin: pd.DataFrame) -> pd.DataFrame:
    columns = np.concatenate((origin["Indices of Deprivation"].unique(), ["FeatureCode"]))
    iodByFeaturecode = pd.DataFrame(columns=columns)
    iodByFeaturecode.set_index("FeatureCode", inplace=True)
    iodByFeaturecode = inputRow(iodByFeaturecode, origin=origin)
    iodByFeaturecode.index.rename("LSOA Code", inplace=True)    
    return iodByFeaturecode.reindex(sorted(iodByFeaturecode.columns), axis=1)



imd2015DF = remove_columns(imd2015DF)
imd2019DF = remove_columns(imd2019DF)

score15, rank15, decile15 = threeWaySplit(df=imd2015DF)

iodByFeatCode15score = preProcessing(origin=score15)
# iodByFeatCode15rank= preProcessing(origin=rank15)
# iodByFeatCode15decile= preProcessing(origin=decile15)

score19, rank19, decile19 = threeWaySplit(df = imd2019DF)

iodByFeatCode19score = preProcessing(origin=score19)
# iodByFeatCode19rank = preProcessing(origin=rank19)
# iodByFeatCode19decile = preProcessing(origin=decile19)

dropnadDBpath = dbPath()

conn = sqlite3.connect(dropnadDBpath)
iodByFeatCode19score.to_sql(
    'IOD_feature_code_score_19',
    conn,
    if_exists='replace',
    index=True
)

iodByFeatCode15score.to_sql(
    'IOD_feature_code_score_15',
    conn,
    if_exists='replace',
    index=True
)
conn.close()
