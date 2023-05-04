from pathlib import Path
import os
from typing import List, Generator, Dict
from tqdm import tqdm
import geopandas as gpd
import pandas as pd
from fiona.drvsupport import supported_drivers
import matplotlib.pyplot as plt
import shapely as shp

import mysql.connector
from mysql.connector.cursor import MySQLCursor



def getForceBoundariesDir() -> Path:
    cwd = Path.cwd()
    parent = cwd.parent
    data = parent.joinpath("data")
    unzipped = data.joinpath("unzipped")
    force_kml = unzipped.joinpath("force_boundaries")
    return force_kml


def loadKMLtoGPDF(path: Path) -> gpd.GeoDataFrame:
    supported_drivers["KML"] = 'rw'
    dF = gpd.read_file(path, driver="KML")
    return dF

def policeForceGDFmaker()->gpd.GeoDataFrame:
    force_KML_path: Path = getForceBoundariesDir()
    force_kml: Generator = force_KML_path.iterdir()
    kmlFileLst: List[str] = os.listdir(force_KML_path)
    regionLst: List[str] = list(map(lambda x: x[:-4], kmlFileLst))
    polygonLst: List[shp.geometry.multipolygon.MultiPolygon] = list(
        map(
            lambda x:
            loadKMLtoGPDF(x)["geometry"][0],
            force_kml
        )
    )

    nameAndShape: Dict = {"name": regionLst, "geometry": polygonLst}

    df = pd.DataFrame.from_dict(nameAndShape)
    gdf = gpd.GeoDataFrame(df, geometry="geometry")
    return gdf

def plotter(gdf: gpd.GeoSeries, save: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(50, 50))
    gdf.plot(column="name", ax=ax, cmap="tab20c", legend=True)
    if save:
        plt.savefig("data/forceRegions.png")


policeForceGDF = policeForceGDFmaker()
policeForceGDF.head()
print("GDF MADE")
policeForceGDF.to_file(getForceBoundariesDir(), driver="GeoJSON")
print("GeoJSON SAVED")
# %%
# mydb = mysql.connector.connect(
#         host="localhost",
#         user="DC2",
#         password="LondonData#641",
#         database='forceRegions'
#         )
# print("CONNECTED TO DATABASE")
# mycursor = mydb.cursor()

# mycursor.execute("DROP TABLE forceKML")
# print("TABLE CLEARED")

# create_table = """
# CREATE TABLE forceKML (
#         name VARCHAR(255) PRIMARY KEY,
#         geom GEOMETRY
#         )
# """
# print("TABLE CREATED")
# mycursor.execute(create_table)


# for index, row in tqdm(policeForceGDF.iterrows()):
#     name = row["name"]
#     geom = row["geometry"].wkt
#     insert_sql = f"INSERT INTO forceKML (name, geom) VALUES('{name}', ST_GeomFromText('{geom}'))"
#     mycursor.execute(insert_sql)

# mydb.commit()
# mydb.close()

#
# %%
