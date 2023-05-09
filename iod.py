from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def getIMD() -> Tuple[pd.DataFrame, pd.DataFrame]:
    cwd = Path.cwd()
    DLUHC_DIR= cwd.joinpath("data/unzipped/DLUHC_open_data")
    imd2015path = DLUHC_DIR.joinpath('imd2015lsoa.csv')
    imd2019path = DLUHC_DIR.joinpath('imd2019lsoa.csv')
    imd2015DF = pd.read_csv(imd2015path)
    imd2019DF = pd.read_csv(imd2019path)
    return imd2015DF, imd2019DF


def cleanIMD(df: pd.DataFrame) -> pd.DataFrame:
    del df["Units"]
    def df["DateCode"]
    return df

def splitIMDframe(df: pd.DataFrame) -> pd.DataFrame:
    scoreFrame = df[df["Measurement"] == "Score"]
    rankFrame = df[df["Measurement"] == "Rank"]
    decileFrame = df[df["Measurement"] == "Decile"]
    return scoreFrame, rankFrame, decileFrame



imd2015DF, imd2019DF = getIMD()




