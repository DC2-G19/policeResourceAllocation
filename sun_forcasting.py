import xgboost as xgb
import pandas as pd
from pathlib import Path
import sqlite3
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error

import matplotlib.pyplot as plt



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

def main():
    conn = sqlite3.connect(dbPath())
    query_sunlight = """
    SELECT *
    FROM sunlight
    """
    sunlightDF = pd.read_sql(query_sunlight, conn)

    sunlightDF["Year-Month"] = pd.to_datetime(sunlightDF["Year-Month"])
    month_dummies = pd.get_dummies(sunlightDF["Year-Month"].dt.month)
    sunlightDF = pd.concat([sunlightDF, month_dummies], axis=1)
    sunlightDF.drop(columns=["index"], inplace=True)
    for i in range(1,12):
        sunlightDF[f"Sunlight_shift_{i}"] = sunlightDF["Sunlight"].shift(i)

    sunlightDF = sunlightDF.set_index(sunlightDF["Year-Month"])
    sunlightDF.drop(columns = ["Year-Month"], inplace=True)

    sunlightDF.dropna(inplace=True)


    X_train, X_test, y_train, y_test = train_test_split(sunlightDF.drop(columns= ["Sunlight"]),sunlightDF["Sunlight"])

    
    model = xgb.XGBRegressor()
    model.fit(X_train.values, y_train)
    model.save_model(modelPath("sunlight"))
    predictions= model.predict(X_test.values)
    pred_train = model.predict(X_train.values)
    plt.scatter(X_test.index, y_test, color="green")
    plt.scatter(X_test.index, predictions, color="red")
    plt.title("Sunlight Forecasting")
    cwd = Path.cwd()
    dc2 = cwd.parent
    img = dc2.joinpath("data/img/")
    plt.savefig(img.joinpath("sunlightForecasting.png"))

    mse_train = mean_squared_error(y_train.values, pred_train)
    mae_train = mean_absolute_error(y_train.values, pred_train)
    r2_train = r2_score(y_train.values, pred_train)
    medae_train = median_absolute_error(y_train.values, pred_train)

# Calculate the test performance metrics
    mse_test = mean_squared_error(y_test.values, predictions)
    mae_test = mean_absolute_error(y_test.values, predictions)
    r2_test = r2_score(y_test.values, predictions)
    medae_test = median_absolute_error(y_test.values, predictions)

# Create a dictionary with the train and test metric names and values
    metrics = {
        'Metric': ['MSE', 'MAE', 'R^2', "MedAE"],
        'Train Value': [mse_train, mae_train, r2_train, medae_train],
        'Test Value': [mse_test, mae_test, r2_test, medae_test]
    }
    metricsDF = pd.DataFrame(metrics)
    print(metricsDF)

if __name__ == "__main__":
    main()

