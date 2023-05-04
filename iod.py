from pathlib import Path
import pandas as pd
from typing import Tuple



def getIMD() -> Tuple[pd.DataFrame, pd.DataFrame]:
    cwd = Path.cwd()
    DLUHC_DIR= cwd.joinpath("data/unzipped/DLUHC_open_data")
    imd2015path = DLUHC_DIR.joinpath('imd2015lsoa.csv')
    imd2019path = DLUHC_DIR.joinpath('imd2019lsoa.csv')
    imd2015DF = pd.read_csv(imd2015path)
    imd2019DF = pd.read_csv(imd2019path)
    return imd2015DF, imd2019DF


imd2015DF, imd2019DF = getIMD()

#%%
del imd2015DF["Units"]
del imd2015DF["DateCode"]
#%%
imd2015DF.describe()

#%%
imd2015DF.head()

#%%
imd2015DF["Measurement"].unique()

#%%
count = imd2015DF["FeatureCode"].value_counts()
print((len(count[count>1]),len(imd2015DF["FeatureCode"].unique())))

#%%
imd2015DF["Indices of Deprivation"].unique()


