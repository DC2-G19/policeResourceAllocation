import pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import statsmodels.api as sm
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

df = pd.read_csv('C:/Users/20212324/DC2/conc.csv')

def plot_crime_type_histogram(df : pandas.DataFrame):
    grouped1 = df.groupby('Crime type', dropna=False).size()
    grouped1.plot.bar(x=grouped1.index, y=grouped1[:], rot=90)
    plt.show()

def correlation_heatmap(df : pandas.DataFrame):
    grouped2 = df.groupby(['LSOA code', 'Crime type']).size().reset_index(name='count')
    pivoted = grouped2.pivot_table(index='LSOA code', columns='Crime type', values='count', fill_value=0)
    new_df = pd.DataFrame(pivoted)
    corr_matrix = new_df.corr(method ='pearson')
    fig, ax = plt.subplots(figsize=(15,10))
    vmin = -max(abs(corr_matrix.values.flatten()))
    vmax = max(abs(corr_matrix.values.flatten()))
    cmap = sns.diverging_palette(-1, 1, as_cmap=True)
    sns.heatmap(corr_matrix, annot=True, cmap=cmap, vmin=vmin, vmax=vmax, center=0)
    plt.show()
# all correlations are between 0.5 and 0.9
# highest are vehicle crime (0.88), criminal damage and arson (0,87), violence and sexual offences (0.84), anti-social behaviour (0.83)

def crime_types_scatter_matrix(df : pandas.DataFrame):
    grouped2 = df.groupby(['LSOA code', 'Crime type']).size().reset_index(name='count')
    pivoted = grouped2.pivot_table(index='LSOA code', columns='Crime type', values='count', fill_value=0)
    new_df = pd.DataFrame(pivoted)
    scatter_matrix(new_df, figsize=(20,20), s=30, hist_kwds={'color':'teal','bins':20, 'alpha':0.5}, alpha=.5)
    plt.show()

def OLS_regression(df : pandas.DataFrame, y_columns : pandas.DataFrame):
    grouped2 = df.groupby(['LSOA code', 'Crime type']).size().reset_index(name='count')
    pivoted = grouped2.pivot_table(index='LSOA code', columns='Crime type', values='count', fill_value=0)
    new_df = pd.DataFrame(pivoted)
    model = sm.OLS(new_df["Burglary"], new_df.drop(["Burglary"], axis=1)).fit()
    print(model.summary())
# new_df.drop(["Burglary"], axis=1):
# all coefficients are significant
# vehicle crime does a lot (also had the highest correlation), Public order and robbery less but are neg and pos respectively.
# Other crime and Other theft and Public disorder and weapons should not be included (also had quite low correlation)

# according to stepwise regression: remove Public disorder and weapons, Other theft, Other crime for sure.

def crimes_line_plot(df : pandas.DataFrame, list_wanted_types : list = list(pd.unique(df['Crime type']))):
    grouped = df.groupby(['Month', 'Crime type']).size()
    unstack_grouped = grouped.unstack('Crime type')
    unstack_grouped = unstack_grouped[list_wanted_types]
    unstack_grouped.plot(kind='line', figsize=(12, 8))
    plt.xlabel('Month')
    plt.ylabel('Number of Crimes')
    plt.title('Number of Crimes per Crime Type per Month')
    plt.show()

def lasso_crime_types(df : pd.DataFrame):
    grouped = df.groupby(['LSOA code', 'Crime type']).size().reset_index(name='count')
    pivoted = grouped.pivot_table(index='LSOA code', columns='Crime type', values='count', fill_value=0)
    new_df = pd.DataFrame(pivoted)
    new_df = new_df.rename_axis(None, axis='columns')
    X = new_df.drop('Burglary', axis=1)
    X = X.reset_index(drop=True)
    features = list(X.columns)
    X.columns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    X = X.to_numpy()
    y = new_df['Burglary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())    ])
    search = GridSearchCV(pipeline,
                          {'model__alpha': np.arange(0.001, 10, 0.1)},
                          cv=5, scoring="neg_mean_squared_error", verbose=3)
    search.fit(X_train, y_train)
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)
    return np.array(features)[importance > 0], np.array(features)[importance == 0]
# useful ['Anti-social behaviour', 'Bicycle theft',
#        'Criminal damage and arson', 'Drugs', 'Other crime', 'Other theft',
#        'Possession of weapons', 'Public order', 'Robbery', 'Shoplifting',
#        'Vehicle crime', 'Violence and sexual offences', 'Violent crime'] (all but unuseful)
# unuseful ['Public disorder and weapons', 'Theft from the person']

def random_forest_feature_selection(df):
    grouped = df.groupby(['LSOA code', 'Crime type']).size().reset_index(name='count')
    pivoted = grouped.pivot_table(index='LSOA code', columns='Crime type', values='count', fill_value=0)
    new_df = pd.DataFrame(pivoted)
    new_df = new_df.rename_axis(None, axis='columns')
    X = new_df.drop('Burglary', axis=1)
    X = X.reset_index(drop=True)
    features = list(X.columns)
    X = X.to_numpy()
    y = new_df['Burglary']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    sel = SelectFromModel(RandomForestClassifier(n_estimators=100), 0.09) # threshold 0.09 for 4 features (higher=less), from 4 it shows stablelt 4 same
    sel.fit(X_train, y_train)
    selected_feat = np.array(features)[(sel.get_support())]

    # plot feature importances
    importances = sel.estimator_.feature_importances_
    importance_series = pd.Series(importances, features)
    fig, ax = plt.subplots()
    importance_series.plot(kind='bar', ax=ax)
    ax.set_xlabel('Features')
    ax.set_ylabel('Importance')
    ax.set_title('Feature Importances')
    plt.xticks(rotation=90)
    plt.show()

    return selected_feat
# choose: Anti-social behaviour, Criminal damage and arson, Vehicle crime, Violence and sexual offences


# USE: Anti-social behaviour, Criminal damage and arson, Vehicle crime, Violence and sexual offences (highest correlation and by random forest)
# NOT: Public disorder and weapons, Theft from the person, Other theft, Other crime


def nb_model(df : pd.DataFrame, vars_list: list):
    print('model: Burglary ~', vars_list)
    # prepare the data
    grouped = df.groupby(['LSOA code', 'Crime type']).size().reset_index(name='count')
    pivoted = grouped.pivot_table(index='LSOA code', columns='Crime type', values='count', fill_value=0)
    new_df = pd.DataFrame(pivoted)
    crime_data = new_df[['Anti-social behaviour', 'Criminal damage and arson', 'Vehicle crime', 'Violence and sexual offences','Burglary']]

    # train the nb model
    X = crime_data[vars_list]
    y = crime_data[['Burglary']]
    X = sm.add_constant(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    model = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial())
    result = model.fit()
    print(result.summary())

    # evaluate the nb model
    nb2_predictions = result.get_prediction(X_test)
    predictions_summary_frame = nb2_predictions.summary_frame()
    squared_residuals = result.resid_response ** 2
    sum_squared_residuals = squared_residuals.sum()
    chi2 = result.pearson_chi2
    deviance = result.deviance
    log_likelihood = result.llf
    print('SSR:', sum_squared_residuals, 'Chi2:', chi2, 'Deviance:', deviance, 'Log-likelihood:', log_likelihood)



