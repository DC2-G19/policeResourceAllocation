import geopandas as gpd
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
from shapely.geometry import Polygon




# map with ward bounds
# prepare the data
cpw = pd.read_csv('C:/Users/20212324/DC2/counts_per_ward.csv') # csv file with counts of burglary per ward in Barnet
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
ax = merged_cpw_geo.plot(column='Count', cmap='Blues', edgecolor='black', linewidth=0.5, figsize=(10, 10))
# for annotations:
for idx, row in merged_cpw_geo.iterrows():
    centroid = row['geometry'].centroid
    ax.annotate( str(row['Ward code']) + "\n" + str(row['Count']), xy=(centroid.x, centroid.y), xytext=(-20, 0), textcoords="offset points", fontsize=8)
# style the map
ax.set_title('Predicted Number of Burglaries per Ward')
sm = plt.cm.ScalarMappable(cmap='Blues')
sm.set_array(merged_cpw_geo['Count'])
cbar = plt.colorbar(sm, orientation='vertical', shrink=0.6, ax=ax)
cbar.set_label('Count')
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()





# plain ward boundaries
#combined_gdf.plot()
#plt.show()






# map with LSOA bounds
# prepare the data
geo_LSOA = gpd.read_file('C:/Users/20212324/DC2/barnet_lsoa.geojson') # LSOA boundaries
geo_LSOA = geo_LSOA[['geometry', 'lsoa11cd']]
geo_LSOA.rename(columns={'lsoa11cd': 'LSOA code'}, inplace=True)
counts_per_LSOA = pd.read_csv('C:/Users/20212324/DC2/LSOA_count.csv') # cvs file with LSOA code and counts
merged_cpl_geo = pd.merge(geo_LSOA, counts_per_LSOA, on=['LSOA code'])
print(merged_cpl_geo)
# Plot the merged GeoDataFrame with filled polygons
ax = merged_cpl_geo.plot(column='Predicted_nr_burglaries', cmap='Blues', edgecolor='black', linewidth=0.5, figsize=(10, 10))
## for annotations:
#for idx, row in merged_cpl_geo.iterrows():
    #centroid = row['geometry'].centroid
    #ax.annotate( str(row['LSOA code']) + "\n" + str(row['Predicted_nr_burglaries']), xy=(centroid.x, centroid.y), xytext=(-20, 0), textcoords="offset points", fontsize=8)
# style the map
ax.set_title('Predicted Number of Burglaries per LSOA')
sm = plt.cm.ScalarMappable(cmap='Blues')
sm.set_array(merged_cpl_geo['Predicted_nr_burglaries'])
cbar = plt.colorbar(sm, orientation='vertical', shrink=0.6, ax=ax)
cbar.set_label('Count')
plt.xlabel('latitude')
plt.ylabel('longitude')
plt.show()






# barplot counts per ward
cpw_sorted = cpw.sort_values(by='Count', ascending=False)
plt.figure(figsize=(10, 6))
plt.bar(cpw_sorted['Ward code'], cpw_sorted['Count'])
plt.xticks(rotation='vertical')
plt.xlabel('Ward code')
plt.ylabel('Count')
plt.title('Counts per Ward')
plt.tight_layout()
plt.show()






# interactive map with wards count
"""import folium
import webbrowser
from branca.colormap import linear
# Create a colormap based on the count values
colormap = linear.YlOrRd_09.scale(merged_cpw_geo['Count'].min(), merged_cpw_geo['Count'].max())
# Create a blank map centered on the first ward polygon
m = folium.Map(location=[merged_cpw_geo['geometry'][0].centroid.y, merged_cpw_geo['geometry'][0].centroid.x], zoom_start=11)
# Add GeoJson layer to the map
folium.GeoJson(
    merged_cpw_geo,
    name='Ward Boundaries',
    style_function=lambda feature: {
        'fillColor': colormap(feature['properties']['Count']),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(fields=['Ward code', 'Count'], labels=True, sticky=True)
).add_to(m)
# Add the colormap to the map
colormap.add_to(m)
# Save the map as an HTML file
map_file = 'interactive_map.html'
m.save(map_file)
# Open the HTML file in the default web browser
webbrowser.open_new_tab(map_file)"""

# interactiva map with ward counts and lsoa points
import folium
import webbrowser
import pandas as pd
from branca.colormap import linear

df = pd.read_csv('C:/Users/20212324/DC2/lat_long_nr.csv') # csv file with a latitude and logitude for each LSOA and the counts of burglary for that
# Merge the ward boundaries data with the count data from the CSV
merged_data = pd.merge(merged_cpw_geo, df, on='Ward code')
# Create a colormap based on the count values
count_colormap = linear.YlOrRd_09.scale(merged_data['Count'].min(), merged_data['Count'].max())
# Create a blank map centered on the first ward polygon
m = folium.Map(location=[merged_cpw_geo['geometry'][0].centroid.y, merged_cpw_geo['geometry'][0].centroid.x], zoom_start=11)
# Add GeoJson layer to the map for ward boundaries
folium.GeoJson(
    merged_data,
    name='Ward Boundaries',
    style_function=lambda feature: {
        'fillColor': count_colormap(feature['properties']['Count']),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(fields=['Ward code', 'Count'], labels=True, sticky=True)
).add_to(m)
# Create a colormap based on the predicted_nr_burglaries values
nr_colormap = linear.YlOrRd_09.scale(df['Predicted_nr_burglaries'].min(), df['Predicted_nr_burglaries'].max())
# Add markers to the map for each location
for index, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color='black',
        fill=True,
        fill_color=nr_colormap(row['Predicted_nr_burglaries']),
        fill_opacity=0.7,
        tooltip=f"LSOA Code: {row['LSOA code']}<br>Predicted Nr. of Burglaries: {row['Predicted_nr_burglaries']}"
    ).add_to(m)
# Add the count colormap to the map
count_colormap.add_to(m)
# Save the map as an HTML file
m.save('interactive_map_ward.html')
# Open the HTML file in the default web browser
#webbrowser.open_new_tab('interactive_map_ward.html')



















# interactiva map with LSOA counts and lsoa points
import folium
import webbrowser
import pandas as pd
from branca.colormap import linear

df = pd.read_csv('C:/Users/20212324/DC2/lat_long_nr.csv') # csv file with a latitude and logitude for each LSOA and the counts of burglary for that
merged_cpl_geo.rename(columns={'Predicted_nr_burglaries': 'Count'}, inplace=True)
# Merge the ward boundaries data with the count data from the CSV
merged_data = pd.merge(merged_cpl_geo, df, on='LSOA code')
# Create a colormap based on the count values
count_colormap = linear.YlOrRd_09.scale(merged_data['Count'].min(), merged_data['Count'].max())
# Create a blank map centered on the first ward polygon
m = folium.Map(location=[merged_cpw_geo['geometry'][0].centroid.y, merged_cpw_geo['geometry'][0].centroid.x], zoom_start=11)
# Add GeoJson layer to the map for ward boundaries
folium.GeoJson(
    merged_data,
    name='Ward Boundaries',
    style_function=lambda feature: {
        'fillColor': count_colormap(feature['properties']['Count']),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(fields=['LSOA code', 'Count'], labels=True, sticky=True)
).add_to(m)
# Create a colormap based on the predicted_nr_burglaries values
nr_colormap = linear.YlOrRd_09.scale(df['Predicted_nr_burglaries'].min(), df['Predicted_nr_burglaries'].max())
# Add markers to the map for each location
for index, row in df.iterrows():
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color='black',
        fill=True,
        fill_color=nr_colormap(row['Predicted_nr_burglaries']),
        fill_opacity=0.7,
        tooltip=f"LSOA Code: {row['LSOA code']}<br>Predicted Nr. of Burglaries: {row['Predicted_nr_burglaries']}"
    ).add_to(m)
# Add the count colormap to the map
count_colormap.add_to(m)
# Save the map as an HTML file
m.save('interactive_map_LSOA.html')
# Open the HTML file in the default web browser
#webbrowser.open_new_tab('interactive_map_LSOA.html')










# interactive map with LSOA and ward boundaries as different layers, no LSOA points
import folium
import webbrowser
import pandas as pd
from branca.colormap import linear

df = pd.read_csv('C:/Users/20212324/DC2/lat_long_nr.csv')
merged_data = pd.merge(merged_cpw_geo, df, on='Ward code')
merged_data2 = pd.merge(merged_cpl_geo, df, on='LSOA code')
count_colormap = linear.YlOrRd_09.scale(merged_data['Count'].min(), merged_data['Count'].max())
m = folium.Map(location=[merged_cpw_geo['geometry'][0].centroid.y, merged_cpw_geo['geometry'][0].centroid.x], zoom_start=11)
folium.GeoJson(
    merged_data,
    name='Ward Boundaries',
    style_function=lambda feature: {
        'fillColor': count_colormap(feature['properties']['Count']),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(fields=['Ward code', 'Count'], labels=True, sticky=True)
).add_to(m)
folium.GeoJson(
    merged_data2,
    name='LSOA Boundaries',
    style_function=lambda feature: {
        'fillColor': count_colormap(feature['properties']['Count']),
        'color': 'black',
        'weight': 0.5,
        'fillOpacity': 0.7,
    },
    tooltip=folium.GeoJsonTooltip(fields=['LSOA code', 'Count'], labels=True, sticky=True)
).add_to(m)
# add marker points to the map for each LSOA
#nr_colormap = linear.YlOrRd_09.scale(df['Predicted_nr_burglaries'].min(), df['Predicted_nr_burglaries'].max())
#for index, row in df.iterrows():
#    folium.CircleMarker(
#        location=[row['Latitude'], row['Longitude']],
#        radius=5,
#        color='black',
#        fill=True,
#        fill_color=nr_colormap(row['Predicted_nr_burglaries']),
#        fill_opacity=0.7,
#        tooltip=f"LSOA Code: {row['LSOA code']}<br>Predicted Nr. of Burglaries: {row['Predicted_nr_burglaries']}"
#    ).add_to(m)
count_colormap.add_to(m)
folium.LayerControl().add_to(m)
map_file = 'interactive_map.html'
m.save(map_file)
webbrowser.open_new_tab(map_file)