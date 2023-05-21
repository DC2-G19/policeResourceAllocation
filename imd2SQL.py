from pathlib import Path
import pandas as pd
import numpy as np
import sqlite3
from typing import Tuple, List, Dict
from tqdm import tqdm


def dbPath() -> Path:
    cwd = Path.cwd()
    parent = cwd.parent
    database = parent.joinpath("data/database_conc_na.db")
    return database


def getIMD() -> Tuple[pd.DataFrame, pd.DataFrame]:
    cwd = Path.cwd()
    parent = cwd.parent
    DLUHC_DIR= parent.joinpath("data/unzipped/DLUHC_open_data")
    imd2015path = DLUHC_DIR.joinpath('imd2015lsoa.csv')
    imd2019path = DLUHC_DIR.joinpath('imd2019lsoa.csv')
    imd2015DF = pd.read_csv(imd2015path)
    imd2019DF = pd.read_csv(imd2019path)
    return imd2015DF, imd2019DF

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



def main():
    imd2015DF, imd2019DF = getIMD()

    imd2015DF = remove_columns(imd2015DF)
    imd2019DF = remove_columns(imd2019DF)

    score15, rank15, decile15 = threeWaySplit(df=imd2015DF)
    print("PREPROCESSING")
    iodByFeatCode15score = preProcessing(origin=score15)
    iodByFeatCode15rank= preProcessing(origin=rank15)
    iodByFeatCode15decile= preProcessing(origin=decile15)

    score19, rank19, decile19 = threeWaySplit(df = imd2019DF)

    iodByFeatCode19score = preProcessing(origin=score19)
    iodByFeatCode19rank = preProcessing(origin=rank19)
    iodByFeatCode19decile = preProcessing(origin=decile19)
    


    dropnadDBpath = dbPath()
    print("CONNECTING TO DATABASE")
    conn = sqlite3.connect(dropnadDBpath)
    print("ADDING SCORE TO DATABASE")
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
    
    print("ADDING RANKING TO DATABASE")

    iodByFeatCode19rank.to_sql(
        'IOD_feature_code_rank_19',
        conn,
        if_exists='replace',
        index=True
    )

    iodByFeatCode15rank.to_sql(
        'IOD_feature_code_rank_15',
        conn,
        if_exists='replace',
        index=True
    )
    print("ADDING DECILE TO DATABASE")
    iodByFeatCode19decile.to_sql(
        'IOD_feature_code_decile_19',
        conn,
        if_exists='replace',
        index=True
    )

    iodByFeatCode15decile.to_sql(
        'IOD_feature_code_decile_15',
        conn,
        if_exists='replace',
        index=True
    )
    conn.close()
    print('CLOSING CONNECTION TO DATABASE')






if __name__ == "__main__":
    main()



