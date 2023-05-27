import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# unemployment data
df_transposed = pd.read_csv('C:/Users/20212324/DC2/monthly_unemployment_rates_london.csv')

# main data
orig_data = pd.read_csv('C:/Users/20212324/DC2/conc_clean.csv')
orig_data_burgl = orig_data[orig_data['Crime type']=='Burglary']
crimes_per_month = orig_data_burgl.groupby('Month').size()
df_crimes = pd.DataFrame({'Month': crimes_per_month.index, 'Number of Crimes': crimes_per_month.values})
df_crimes.set_index('Month', inplace=True)

# plot unemployment rate and burglary
fig, ax1 = plt.subplots(figsize=(20, 10))
df_crimes.plot(kind='line', ax=ax1)

ax1.set_title('Group Counts')
ax1.set_xlabel('Date')
ax1.set_ylabel('Burglary Count')

df_transposed.index = pd.to_datetime(df_transposed.index).strftime('%Y-%m')
ax2 = ax1.twinx()  # Create a twin Axes sharing the x-axis
ax2.plot(df_transposed.index, df_transposed, color='red')
ax2.set_ylabel('London Unemployment Rate', color='red')

# Reduce the number of dates displayed on x-axis of ax2
tick_locations = ax2.get_xticks()
subset_ticks = tick_locations[::3]  # Display every other tick for ::2

ax2.set_xticks(subset_ticks)
ax1.xaxis.set_tick_params(rotation=90)

plt.show()

def OLS(df_crimes, df_unemployment):
    ## Convert the columns to numeric data types
    df_crimes["Number of Crimes"] = pd.to_numeric(df_crimes["Number of Crimes"])
    df_unemployment['Unemployment Rate'] = pd.to_numeric(df_unemployment['Unemployment Rate'])

    # make sure the indices are the same
    df_combined = pd.concat([df_crimes, df_unemployment], axis=1, join='inner')
    y = df_combined["Number of Crimes"]
    x = df_combined['Unemployment Rate']

    model = sm.OLS(y, x).fit()
    return model.summary()


# scatterplot, clustering
df_combined = pd.concat([df_crimes, df_transposed], axis=1, join='inner')
y = df_combined["Number of Crimes"]
x = df_combined['Unemployment Rate']

kmeans = KMeans(n_clusters=3)
kmeans.fit(df_combined)

labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_
num_clusters = len(cluster_centers)
colors = ['red', 'blue', 'green', 'orange', 'purple', 'yellow']

# Plot the data points with different colors for each cluster
for i in range(len(df_combined)):
    cluster_label = labels[i]
    plt.scatter(df_combined.iloc[i, 0], df_combined.iloc[i, 1], c=colors[cluster_label])

# Plot the cluster centers
for i in range(num_clusters):
    plt.scatter(cluster_centers[i, 0], cluster_centers[i, 1], c='black', marker='x')

plt.xlabel('Number of Crimes')
plt.ylabel('Unemployment Rate')
plt.title('K-means Clustering')

plt.show()