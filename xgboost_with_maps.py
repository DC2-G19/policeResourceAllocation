import sqlite3
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor
from pathFunc import dbPath, dataDir

conn = sqlite3.connect(dbPath())
conn = sqlite3.connect(path)
query_burg = """SELECT * FROM table_name
WHERE "Crime Type" = "Burglary"
"""
query_sun = """SELECT * FROM sunlight"""
query_house = """SELECT * FROM housing_by_lsoa"""
query_unep = """SELECT * FROM montly_unemployement_claimant_count_by_lsoa_barnet"""

df_burglary_all = pd.read_sql(query_burg, conn)
df_sunlight_all = pd.read_sql(query_sun, conn)
df_housing_all = pd.read_sql(query_house, conn)
df_unemployment_all = pd.read_sql(query_unep, conn)
#getting data from database and storing in DF
conn.close()

df_unemployment = df_unemployment_all.rename(columns={'geogcode': 'LSOA code'})
df_unemployment['Month'] =  pd.to_datetime(df_unemployment['date'])
df_unemployment_5yr = df_unemployment[(df_unemployment['Month'].dt.year >= 2013) & (df_unemployment['Month'].dt.year <= 2019)]
df_unemployment_5yr = df_unemployment_5yr[df_unemployment_5yr['LSOA code'] != 'Column Total']
df_unemployment_5yr["Unemployment"] = df_unemployment_5yr["value"]
df_unemployment_5yr = df_unemployment_5yr.drop(columns=["date", "index", "value"], axis=1)
#filterig unemployment to 5 year time spand of 2014 to 2019 and renaming columns for compatibility
#2013 included for time lag
#%%

df_bar=df_burglary_all[df_burglary_all['LSOA name'].str.contains('Barnet')].drop(columns=['Reported by', 'Falls within']).dropna()
df_bar['Month'] = pd.to_datetime(df_bar['Month'])
df_bar['Year']= df_bar['Month'].dt.year
df_bar_5yr = df_bar[(df_bar['Month'].dt.year >= 2013) & (df_bar['Month'].dt.year <= 2019)]

#Filtering for Barnet data, 2013 included for time lag
#%%
df_sunlight_all['Month'] = pd.to_datetime(df_sunlight_all['Year-Month'])
df_sunlight=df_sunlight_all.drop(columns=["Year-Month", "index"], axis=1)
#Sunlight DF, ensuring compatibility by creating Datime Col

#%%
#df_housing = df_housing.drop(columns=["index"], axis=1)
#Housing dataframe
df_housing_clean = df_housing_all[["Proportion of indep", "Proportion small house", "Proportion social", "Proportion rented", "LSOA name",  "LSOA code" ]]
#%%
burglaries_per_lsoa_month = df_bar_5yr[df_bar_5yr['Crime type'] == 'Burglary'].groupby(['LSOA code', 'Month']).size()
burglaries_per_lsoa_month = burglaries_per_lsoa_month.reset_index(name='Total Burglaries')
#Getting total burglaries per LSOA on a Monthly basis
#%%
temp_merged = burglaries_per_lsoa_month.merge(df_housing_clean, how='left', on=['LSOA code']).dropna() #Merging Berg and Housing
df_merged = temp_merged.merge(df_sunlight, how='left', on=['Month']) #Merging sunlight and (Housing+Burg)
df_merged_2 =  df_merged.merge(df_unemployment_5yr, how='left', on=['Month', 'LSOA code']) #Mergving pervious with unemployment
temp_merged_clean = df_merged_2.drop(columns=['LSOA name']) #Drop unnecessary Cols
#%%
lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  # Giving Lag value in months
df = temp_merged_clean

for lag in lags:
    column_name_unep = f"Unemployment_PctChange_{lag}m"  # New column name for unemployment
    df[column_name_unep] = df.groupby('LSOA code')['Unemployment'].pct_change(periods=lag) * 100
    df[column_name_unep].replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf values with NaN

    column_name_burg = f"Total_Burglaries_PctChange_{lag}m"  # New column name for burglaries
    df[column_name_burg] = df.groupby('LSOA code')['Total Burglaries'].pct_change(periods=lag) * 100
    df[column_name_burg].replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace inf values with NaN

df.fillna(0, inplace=True)  # Replace remaining NaN values with zeros

#%%
# Create dummy variables for the 'LSOA code' column
dummy_df = pd.get_dummies(temp_merged_clean['LSOA code'], prefix='LSOA')

# Concatenate the dummy variables with the original DF
df_temp_merged_clean_dummies = pd.concat([temp_merged_clean, dummy_df], axis=1)
df_temp_merged_clean_dummies['Month_Int'] = df_temp_merged_clean_dummies['Month'].dt.month #Creating month int col
df_temp_merged_clean_dummies.head()

# Create dummy variables for the 'Month_Int' column
month_dummies = pd.get_dummies(df_temp_merged_clean_dummies['Month_Int'], prefix='Month')

# Concatenate the dummy variables with the original DataFrame
df_temp_merged_clean_dummies = pd.concat([df_temp_merged_clean_dummies, month_dummies], axis=1)
#df_temp_merged_clean_dummies.head()

#%%
df_temp_merged_clean_train = df_temp_merged_clean_dummies[(df_temp_merged_clean_dummies['Month'].dt.year >= 2014) & (df_temp_merged_clean_dummies['Month'].dt.year <= 2018)]
df_temp_merged_clean_test = df_temp_merged_clean_dummies[(df_temp_merged_clean_dummies['Month'].dt.year == 2019)]
#Filtering test and train based on time span of 2014 to 2018 for Train and 2019 for Test
#%%
lags_unep_burg = ["Unemployment_PctChange_1m", "Unemployment_PctChange_2m", "Unemployment_PctChange_3m", "Unemployment_PctChange_4m", "Unemployment_PctChange_5m", "Unemployment_PctChange_6m", "Unemployment_PctChange_7m", "Unemployment_PctChange_8m", "Unemployment_PctChange_9m", "Unemployment_PctChange_10m", "Unemployment_PctChange_11m", "Unemployment_PctChange_12m", "Total_Burglaries_PctChange_1m", "Total_Burglaries_PctChange_2m", "Total_Burglaries_PctChange_3m", "Total_Burglaries_PctChange_4m", "Total_Burglaries_PctChange_5m", "Total_Burglaries_PctChange_6m", "Total_Burglaries_PctChange_7m", "Total_Burglaries_PctChange_8m", "Total_Burglaries_PctChange_9m", "Total_Burglaries_PctChange_10m", "Total_Burglaries_PctChange_11m", "Total_Burglaries_PctChange_12m"]

X_train = df_temp_merged_clean_train.drop(columns=["Total Burglaries", "LSOA code", "Month_Int", "Month"]) # What you give to the model
y_train = df_temp_merged_clean_train[["Total Burglaries"]] # What needs to be predicted

X_test = df_temp_merged_clean_test.drop(columns=["Total Burglaries", "LSOA code", "Month_Int", "Month"]) # What you give to the model
y_test = df_temp_merged_clean_test[["Total Burglaries"]] # What needs to be predicted

#%%
X_train.head()
#%%
model = XGBRegressor()
model.fit(X_train, y_train)
#Fitting data to model
#%%
import matplotlib.pyplot as plt
xgb.plot_importance(model, ax=plt.gca(), max_num_features=25)
#Plotting the most important featurs
#%%
pred_train = model.predict(X_train)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, median_absolute_error
predictions = model.predict(X_test)
#%%
# Calculate the train performance metrics
mse_train = mean_squared_error(y_train, pred_train)
mae_train = mean_absolute_error(y_train, pred_train)
r2_train = r2_score(y_train, pred_train)
medae_train = median_absolute_error(y_train, pred_train)

# Calculate the test performance metrics
mse_test = mean_squared_error(y_test, predictions)
mae_test = mean_absolute_error(y_test, predictions)
r2_test = r2_score(y_test, predictions)
medae_test = median_absolute_error(y_test, predictions)

# Create a dictionary with the train and test metric names and values
metrics = {
    'Metric': ['MSE', 'MAE', 'R^2', "MedAE"],
    'Train Value': [mse_train, mae_train, r2_train, medae_train],
    'Test Value': [mse_test, mae_test, r2_test, medae_test]
}

# Create a DataFrame from the dictionary
metrics_df = pd.DataFrame(metrics)
metrics_df
#%%
df_date_lsoa_burg_temp = df_temp_merged_clean_test[['LSOA code', 'Month', 'Total Burglaries']].reset_index(drop=True, inplace=False)
predictions_df = pd.DataFrame(predictions, columns=['Prediction'])
predictions_df[['LSOA code', 'Month', 'Total Burglaries']] = df_date_lsoa_burg_temp
print(predictions_df)

# add ward codes
df_w = pd.read_csv('C:/Users/20212324/DC2/Lower_Layer_Super_Output_Area_(2021)_to_Ward_(2023)_to_LAD_(2023)_Lookup_in_England_and_Wales.csv')
# from: https://geoportal.statistics.gov.uk/search?collection=Dataset&q=Lower%20Layer%20Super%20Output%20Area%20(2021)%20to%20Ward%20(2023)
df_ward_LSOA = pd.DataFrame()
df_ward_LSOA[['LSOA code', 'Ward code']] = df_w[['LSOA21CD', 'WD23CD']]
new_df = pd.merge(predictions_df, df_ward_LSOA, on='LSOA code', how='left')
print(new_df)









# map interaction (total number burglaries for the year)

import geopandas as gpd
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

# map with ward bounds
# prepare the data
cpw = new_df.groupby('Ward code')['Prediction'].sum().reset_index()
# combine all ward geojson files into one
geojson_directory = 'C:/Users/20212324/DC2/metropolitan/barnet_wards_geojson/'  # folder with geojson files per ward in Barnet
gdf_list = []
for filepath in glob.glob(os.path.join(geojson_directory, '*.geojson')):
    gdf = gpd.read_file(filepath)
    filename = os.path.basename(filepath)
    ward_code = os.path.splitext(filename)[0]
    gdf['Ward code'] = ward_code
    gdf_list.append(gdf)
combined_gdf = gpd.GeoDataFrame(pd.concat(gdf_list, ignore_index=True))
merged_cpw_geo = pd.merge(combined_gdf, cpw, on=['Ward code'])
# Convert LineString to Polygon
merged_cpw_geo['geometry'] = merged_cpw_geo['geometry'].apply(lambda x: Polygon(x.coords) if x.geom_type == 'LineString' else x)
# Plot the merged GeoDataFrame with filled polygons
ax = merged_cpw_geo.plot(column='Prediction', cmap='Blues', edgecolor='black', linewidth=0.5, figsize=(10, 10))
# for annotations:
for idx, row in merged_cpw_geo.iterrows():
    centroid = row['geometry'].centroid
    ax.annotate( str(row['Ward code']) + "\n" + str(round(row['Prediction'], 1)), xy=(centroid.x, centroid.y), xytext=(-20, 0), textcoords="offset points", fontsize=8)
# style the map
ax.set_title('Predicted Number of Burglaries per Ward')
sm = plt.cm.ScalarMappable(cmap='Blues')
sm.set_array(merged_cpw_geo['Prediction'])
cbar = plt.colorbar(sm, orientation='vertical', shrink=0.6, ax=ax)
cbar.set_label('Prediction')
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()


# plain ward boundaries
combined_gdf.plot()
plt.show()


# map with LSOA bounds
# prepare the data
geo_LSOA = gpd.read_file('C:/Users/20212324/DC2/barnet_lsoa.geojson') # LSOA boundaries
geo_LSOA = geo_LSOA[['geometry', 'lsoa11cd']]
geo_LSOA.rename(columns={'lsoa11cd': 'LSOA code'}, inplace=True)
counts_per_LSOA = new_df.groupby('LSOA code')['Prediction'].sum().reset_index()
merged_cpl_geo = pd.merge(geo_LSOA, counts_per_LSOA, on=['LSOA code'])
# Plot the merged GeoDataFrame with filled polygons
ax = merged_cpl_geo.plot(column='Prediction', cmap='Blues', edgecolor='black', linewidth=0.5, figsize=(10, 10))
## for annotations:
#for idx, row in merged_cpl_geo.iterrows():
    #centroid = row['geometry'].centroid
    #ax.annotate( str(row['LSOA code']) + "\n" + str(row['Prediction']), xy=(centroid.x, centroid.y), xytext=(-20, 0), textcoords="offset points", fontsize=8)
# style the map
ax.set_title('Predicted Number of Burglaries per LSOA')
sm = plt.cm.ScalarMappable(cmap='Blues')
sm.set_array(merged_cpl_geo['Prediction'])
cbar = plt.colorbar(sm, orientation='vertical', shrink=0.6, ax=ax)
cbar.set_label('Prediction')
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()


# barplot counts per ward
cpw_sorted = cpw.sort_values(by='Prediction', ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(cpw_sorted['Ward code'], cpw_sorted['Prediction'])
plt.xticks(rotation='vertical')
plt.xlabel('Ward code')
plt.ylabel('Prediction')
plt.title('Counts per Ward')
plt.tight_layout()
plt.show()



# compare with actual values
# prepare the data
counts_per_LSOA = new_df.groupby('LSOA code')['Total Burglaries'].sum().reset_index()
merged_cpl_geo2 = pd.merge(geo_LSOA, counts_per_LSOA, on=['LSOA code'])
print(merged_cpl_geo2)
# Plot the merged GeoDataFrame with filled polygons
ax = merged_cpl_geo2.plot(column='Total Burglaries', cmap='Blues', edgecolor='black', linewidth=0.5, figsize=(10, 10))
ax.set_title('Actual Number of Burglaries per LSOA')
sm = plt.cm.ScalarMappable(cmap='Blues')
sm.set_array(merged_cpl_geo2['Total Burglaries'])
cbar = plt.colorbar(sm, orientation='vertical', shrink=0.6, ax=ax)
cbar.set_label('Total Burglaries')
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()



# interactive map with LSOA and ward boundaries as different layers
import folium
import webbrowser
import pandas as pd
from branca.colormap import linear

merged_data = merged_cpw_geo[['geometry', 'Prediction', 'Ward code']]
merged_data2 = merged_cpl_geo[['geometry', 'Prediction', 'LSOA code']]
count_colormap = linear.YlOrRd_09.scale(merged_data['Prediction'].min(), merged_data['Prediction'].max())
m = folium.Map(location=[merged_cpw_geo['geometry'][0].centroid.y, merged_cpw_geo['geometry'][0].centroid.x], zoom_start=11)
folium.GeoJson(
    merged_data,
    name='Ward Boundaries',
    style_function=lambda feature: {
        'fillColor': count_colormap(feature['properties']['Prediction']),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(fields=['Ward code', 'Prediction'], labels=True, sticky=True)
).add_to(m)
folium.GeoJson(
    merged_data2,
    name='LSOA Boundaries',
    style_function=lambda feature: {
        'fillColor': count_colormap(feature['properties']['Prediction']),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(fields=['LSOA code', 'Prediction'], labels=True, sticky=True)
).add_to(m)
count_colormap.add_to(m)
folium.LayerControl().add_to(m)
map_file = 'interactive_map.html'
m.save(map_file)
#webbrowser.open_new_tab(map_file)

































# map interaction (total number burglaries for the month)

# predicted burglaries per ward per month
cpw = new_df.groupby(['Ward code', 'Month'])['Prediction'].sum().reset_index()
merged_cpw_geo = pd.merge(combined_gdf, cpw, on=['Ward code'])
# Convert LineString to Polygon
merged_cpw_geo['geometry'] = merged_cpw_geo['geometry'].apply(lambda x: Polygon(x.coords) if x.geom_type == 'LineString' else x)
# Define the minimum and maximum values for the color scale
vmin = merged_cpw_geo['Prediction'].min()
vmax = merged_cpw_geo['Prediction'].max()
# Loop over each unique month
unique_months = merged_cpw_geo['Month'].unique()
for month in unique_months:
    month_data = merged_cpw_geo[merged_cpw_geo['Month'] == month]
    ax = month_data.plot(column='Prediction', cmap='Blues', edgecolor='black', linewidth=0.5, figsize=(10, 10),
                         vmin=vmin, vmax=vmax)
    # Add annotations for each ward
    for idx, row in month_data.iterrows():
        centroid = row['geometry'].centroid
        ax.annotate(str(row['Ward code']) + "\n" + str(round(row['Prediction'], 1)), xy=(centroid.x, centroid.y),
                    xytext=(-20, 0), textcoords="offset points", fontsize=8)
    # Style the map
    ax.set_title('Predicted Number of Burglaries per Ward - {}'.format(month))
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array(month_data['Prediction'])
    cbar = plt.colorbar(sm, orientation='vertical', shrink=0.6, ax=ax)
    cbar.set_label('Prediction')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.show()



# predicted buglaries per lsoa per month
counts_per_LSOA_month = new_df.groupby(['LSOA code', 'Month'])['Prediction'].sum().reset_index()
unique_months = counts_per_LSOA_month['Month'].unique()
vmin = counts_per_LSOA_month['Prediction'].min()
vmax = counts_per_LSOA_month['Prediction'].max()
for month in unique_months:
    month_data = counts_per_LSOA_month[counts_per_LSOA_month['Month'] == month]
    merged_cpl_geo = pd.merge(geo_LSOA, month_data, on='LSOA code')
    ax = merged_cpl_geo.plot(column='Prediction', cmap='Blues', edgecolor='black', linewidth=0.5, figsize=(10, 10))
    # Add annotations for each LSOA
    #for idx, row in merged_cpl_geo.iterrows():
    #    centroid = row['geometry'].centroid
    #    ax.annotate(str(row['LSOA code']) + "\n" + str(round(row['Prediction'], 1)), xy=(centroid.x, centroid.y), xytext=(-20, 0), textcoords="offset points", fontsize=8)
    # Style the map
    ax.set_title('Predicted Number of Burglaries per LSOA - {}'.format(month))
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array(merged_cpl_geo['Prediction'])
    cbar = plt.colorbar(sm, orientation='vertical', shrink=0.6, ax=ax)
    cbar.set_label('Prediction')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.show()




# actual values per month per LSOA
counts_per_LSOA_month2 = new_df.groupby(['LSOA code', 'Month'])['Total Burglaries'].sum().reset_index()
unique_months2 = counts_per_LSOA_month2['Month'].unique()
vmin = counts_per_LSOA_month['Prediction'].min()
vmax = counts_per_LSOA_month['Prediction'].max()
for month in unique_months2:
    month_data = counts_per_LSOA_month2[counts_per_LSOA_month2['Month'] == month]
    merged_cpl_geo = pd.merge(geo_LSOA, month_data, on='LSOA code')
    ax = merged_cpl_geo.plot(column='Total Burglaries', cmap='Blues', edgecolor='black', linewidth=0.5, figsize=(10, 10))
    # Add annotations for each LSOA
    #for idx, row in merged_cpl_geo.iterrows():
    #    centroid = row['geometry'].centroid
    #    ax.annotate(str(row['LSOA code']) + "\n" + str(round(row['Total Burglaries'], 1)), xy=(centroid.x, centroid.y), xytext=(-20, 0), textcoords="offset points", fontsize=8)
    # Style the map
    ax.set_title('Actual Number of Burglaries per LSOA - {}'.format(month))
    sm = plt.cm.ScalarMappable(cmap='Blues', norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array(merged_cpl_geo['Total Burglaries'])
    cbar = plt.colorbar(sm, orientation='vertical', shrink=0.6, ax=ax)
    cbar.set_label('Total Burglaries')
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.show()



# interactive map for specific month
merged_data = merged_cpw_geo[['geometry', 'Prediction', 'Ward code', 'Month']]
counts_per_LSOA_month = new_df.groupby(['LSOA code', 'Month'])['Prediction'].sum().reset_index()
merged_cpl_geo = pd.merge(geo_LSOA, counts_per_LSOA_month, on=['LSOA code'])
merged_data2 = merged_cpl_geo[['geometry', 'Prediction', 'LSOA code', 'Month']]
month_nr = input('Enter the month number (__): ')
selected_month = '2019-' + month_nr + '-01'
# Filter the data for the selected month on both Ward and LSOA layers
merged_data_selected_month_ward = merged_data[merged_data['Month'] == selected_month]
merged_data_selected_month_lsoa = merged_data2[merged_data2['Month'] == selected_month]
# color maps
ward_min_pred = merged_data_selected_month_ward['Prediction'].min()
ward_max_pred = merged_data_selected_month_ward['Prediction'].max()
lsoa_min_pred = merged_data_selected_month_lsoa['Prediction'].min()
lsoa_max_pred = merged_data_selected_month_lsoa['Prediction'].max()
ward_count_colormap = linear.YlOrRd_09.scale(ward_min_pred, ward_max_pred)
lsoa_count_colormap = linear.YlOrRd_09.scale(lsoa_min_pred, lsoa_max_pred)
# Create the interactive map
m = folium.Map(
    location=[merged_cpw_geo['geometry'][0].centroid.y, merged_cpw_geo['geometry'][0].centroid.x],
    zoom_start=11
)

# Add the Ward boundaries to the map for the selected month
folium.GeoJson(
    merged_data_selected_month_ward.drop(columns=['Month']),
    name='Ward Boundaries',
    style_function=lambda feature: {
        'fillColor': ward_count_colormap(feature['properties']['Prediction']),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(fields=['Ward code', 'Prediction'], labels=True, sticky=True)
).add_to(m)

# Add the LSOA boundaries to the map for the selected month
folium.GeoJson(
    merged_data_selected_month_lsoa.drop(columns=['Month']),
    name='LSOA Boundaries',
    style_function=lambda feature: {
        'fillColor': lsoa_count_colormap(feature['properties']['Prediction']),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(fields=['LSOA code', 'Prediction'], labels=True, sticky=True)
).add_to(m)
ward_count_colormap.caption = 'Ward Predictions'
ward_count_colormap.add_to(m)
lsoa_count_colormap.caption = 'LSOA Predictions'
lsoa_count_colormap.add_to(m)
folium.LayerControl().add_to(m)
map_file = 'interactive_map.html'
m.save(map_file)
webbrowser.open_new_tab(map_file)