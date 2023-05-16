import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

connection = sqlite3.connect('C:/Users/20212324/DC2/database_conc_na.db')
cursor = connection.cursor()
connection_w = sqlite3.connect('C:/Users/20212324/DC2/LSOA_to_Ward.db')
cursor2 = connection_w.cursor()

def create_joined_ward_table():

    attach_query = "ATTACH DATABASE 'LSOA_to_Ward.db' AS LSOA_to_Ward"
    cursor.execute(attach_query)

    # Execute the join query
    join_query = '''
    SELECT *
    FROM table_name
    JOIN LSOA_to_Ward.LSOA_to_Ward ON table_name."LSOA code" = LSOA_to_Ward.LSOA_to_Ward."LSOA21CD";
    '''

    # Execute the query on db1
    cursor.execute(join_query)

    joined = pd.read_sql_query(join_query, connection)
    joined.to_sql('joined_table', connection, if_exists='replace', index=False)

    detach_query = "DETACH DATABASE LSOA_to_Ward"
    cursor.execute(detach_query)

def max_nr_crimes():
    query = '''
        SELECT WD23NM, count("Crime ID")
        FROM joined_table
        WHERE LAD23NM='Barnet'
        GROUP BY WD23NM
        ORDER BY count("Crime ID");
    '''
    df_nr_crimes = pd.read_sql_query(query, connection)
    print(df_nr_crimes)

    # plot the result
    df_sorted = df_nr_crimes.sort_values('count("Crime ID")', ascending=False)
    plt.figure(figsize=(10, 8))
    plt.barh(df_sorted['WD23NM'], df_sorted['count("Crime ID")'])
    plt.xlabel('Crime Count')
    plt.ylabel('Ward')
    plt.title('Number of Crimes per Ward')
    plt.gca().invert_yaxis()

    # Add count labels to the bars
    for i, count in enumerate(df_sorted['count("Crime ID")']):
        plt.text(count + 10, i, str(count), va='center')

    plt.show()

    max_ward = df_nr_crimes[-1:]['WD23NM'].values[0]
    print('ward with most crimes: ' + max_ward)

    # find the nr crimes for the max_ward
    query = '''
        SELECT [LSOA code], [LSOA name], Longitude, Latitude, count("Crime ID")
        FROM joined_table
        WHERE LAD23NM='Barnet' and WD23NM=?
        GROUP BY [LSOA code]
        ORDER BY count("Crime ID");
    '''
    df_most_crimes = pd.read_sql_query(query, connection, params=(max_ward,))
    print(df_most_crimes)

    # plot the result
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_most_crimes['Longitude'], df_most_crimes['Latitude'], s=df_most_crimes['count("Crime ID")'], c=df_most_crimes['count("Crime ID")'], alpha=0.5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Number of crimes per LSOA Location in ' + max_ward)

    # Add LSOA code and count labels
    for i, row in df_most_crimes.iterrows():
        label = "LSOA: " + str(row['LSOA code']) + "\nCount: " + str(row['count("Crime ID")'])
        plt.text(row['Longitude'], row['Latitude'], label, fontsize=8)

    plt.colorbar(scatter)
    plt.axis('equal')

    plt.show()

    print('total number of crimes: ' + str(sum(df_most_crimes['count("Crime ID")'])) + ' percentage from total number of crimes: ' + str((sum(df_most_crimes['count("Crime ID")'])/sum(df_nr_crimes['count("Crime ID")']))*100) + '%')

def max_nr_burglaries():
    query = '''
        SELECT WD23NM, count("Crime ID")
        FROM joined_table
        WHERE LAD23NM='Barnet' and [Crime type]='Burglary'
        GROUP BY WD23NM
        ORDER BY count("Crime ID");
    '''
    df_nr_burglaries = pd.read_sql_query(query, connection)
    print(df_nr_burglaries)

    # plot the result
    df_sorted = df_nr_burglaries.sort_values('count("Crime ID")', ascending=False)
    plt.figure(figsize=(10, 8))
    plt.barh(df_sorted['WD23NM'], df_sorted['count("Crime ID")'])
    plt.xlabel('Crime Count')
    plt.ylabel('Ward')
    plt.title('Number of Burglaries per Ward')
    plt.gca().invert_yaxis()

    # Add count labels to the bars
    for i, count in enumerate(df_sorted['count("Crime ID")']):
        plt.text(count + 10, i, str(count), va='center')

    plt.show()

    max_ward = df_nr_burglaries[-1:]['WD23NM'].values[0]
    print('ward with most burglaries: ' + max_ward)

    # find the nr crimes for the max_ward
    query = '''
        SELECT [LSOA code], [LSOA name], Longitude, Latitude, count("Crime ID")
        FROM joined_table
        WHERE LAD23NM='Barnet' and [Crime type]='Burglary' and WD23NM=?
        GROUP BY [LSOA code]
        ORDER BY count("Crime ID");
    '''
    df_most_burglaries = pd.read_sql_query(query, connection, params=(max_ward,))
    print(df_most_burglaries)

    # plot the reuslt
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(df_most_burglaries['Longitude'], df_most_burglaries['Latitude'], s=df_most_burglaries['count("Crime ID")'], c=df_most_burglaries['count("Crime ID")'], alpha=0.5)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Number of burglaries per LSOA Location in ' + max_ward)

    # Add LSOA code and count labels
    for i, row in df_most_burglaries.iterrows():
        label = "LSOA: " + str(row['LSOA code']) + "\nCount: " + str(row['count("Crime ID")'])
        plt.text(row['Longitude'], row['Latitude'], label, fontsize=8, ha='left')

    plt.colorbar(scatter)
    plt.axis('equal')

    plt.show()

    sum(df_most_burglaries['count("Crime ID")'])
    print('total number of burglaries: ' + str(sum(df_most_burglaries['count("Crime ID")'])) + ' percentage from total number of burglaries: ' + str((sum(df_most_burglaries['count("Crime ID")']) / sum(df_nr_burglaries['count("Crime ID")'])) * 100) + '%' + ' percentage from total number of crimes: ' + str((sum(df_most_burglaries['count("Crime ID")']) /273341) * 100) + '%')


max_nr_crimes()
max_nr_burglaries()
