import xgboost as xgb
import pandas as pd
import sqlite3
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import median_absolute_error, mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

def dbPath() -> Path:
    cwd = Path.cwd()
    dc2 = cwd.parent
    db = dc2.joinpath("data/database_final.db")
    return db

def modelPath(lsoaCode: str) -> Path:
    cwd = Path.cwd()
    parent = cwd.parent
    return parent.joinpath(f"data/models/{lsoaCode}.bin")

def main():
    conn = sqlite3.connect(dbPath())
    query_lsoa_codes = "SELECT geogcode FROM lsoa_code_to_name"
    lsoa_codes = pd.read_sql(query_lsoa_codes, conn)

    query_unemployement = """
    SELECT * 
    FROM montly_unemployement_claimant_count_by_lsoa_barnet
    """
    unemployement = pd.read_sql(query_unemployement, conn)
    unemployement.dropna(inplace=True)
    unemployement.drop(columns="index", inplace=True)
    unemployement = unemployement[(unemployement["date"]<"2020")&(unemployement["date"]>"2012")]
    unemployement = unemployement[unemployement["geogcode"]!="Column Total"].copy()
    
    lsoa_code_list = lsoa_codes[lsoa_codes["geogcode"] != "Column Total"]["geogcode"].to_list()
    allFeatures = pd.DataFrame(columns=lsoa_code_list, index=unemployement["date"].unique())
    
    for row in tqdm(unemployement.index):
        allFeatures[unemployement["geogcode"][row]][unemployement["date"][row]] = unemployement["value"][row]

    for code in tqdm(lsoa_code_list):
        for i in range(12):
            allFeatures[f"{code}_shift_{i+1}"] = allFeatures[f"{code}"].shift(i+1)

    allFeatures = allFeatures.dropna()

    code_out_shift = allFeatures[lsoa_code_list].copy()
    code_with_shift = allFeatures.drop(lsoa_code_list, axis=True).copy()
    
    metrics = {}
    for col in tqdm(lsoa_code_list):
        y = code_out_shift[col].copy()
        x = code_with_shift.copy()
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        model = xgb.XGBRegressor()
        model.fit(X_train.values, y_train)
        y_pred = model.predict(X_test.values)
        mse_test = mean_squared_error(y_test, y_pred)
        mae_test = mean_absolute_error(y_test, y_pred)
        r2_test = r2_score(y_test, y_pred)
        medae_test = median_absolute_error(y_test, y_pred)
        metrics[col] = [mse_test, mae_test, r2_test, medae_test]
        model.save_model( modelPath(col))

    vioplotDF = pd.DataFrame(columns = ["Metric", "Value", "LSOA"])

    for key, val in metrics.items():
        # vioplotDF = vioplotDF.append({"Metric": "MSE", "Value": val[0], "LSOA": key}, ignore_index=True)
        vioplotDF = vioplotDF.append({"Metric": "MAE", "Value": val[1], "LSOA": key}, ignore_index=True)
        vioplotDF = vioplotDF.append({"Metric": "R^2", "Value": val[2], "LSOA": key}, ignore_index=True)
        vioplotDF = vioplotDF.append({"Metric": "MedAE", "Value": val[3], "LSOA": key}, ignore_index=True)
    
    fig, ax = plt.subplots()
    sns.violinplot(data=vioplotDF, x="Metric", y= "Value", ax=ax)
    ax.set_title("Unemployement Forecasting By LSOA Performance")
    cwd = Path.cwd()
    dc2 = cwd.parent
    img = dc2.joinpath("data/img/")
    plt.savefig(img.joinpath("unemployementForecasting.png"))


if __name__ == "__main__":
    main()
