# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 15:29:55 2019

@author: sergi
"""
# P A N D A S and N U M P Y
import numpy as np
import pandas as pd

# S K L E A R N
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import dill

# Regular Expression
import re
#Load data to Pandas

#datafile_path = './Combined_Output_fileComplete NSW Blacktown areas'
pickle_file_path = './RandomForest_Sellable.pkl'

#Load data to Pandas
def load_data():
    df = pd.read_csv('./Combined_Output_fileComplete NSW Blacktown areas.csv')
    df[['Valuation']] = df[['Valuation']].replace(' ', np.nan, regex=True)
    df[['Valuation']] = df[['Valuation']].replace('[!a-zA-Z+&,.\$£/:)(]', '', regex=True)
    df['Valuation'] = pd.to_numeric(df['Valuation'], errors='coerce').fillna(0).astype(np.int64)

#    df['Valuation'] = pd.to_numeric(df['Valuation'])
    return df

# Data Cleaning and EDA
def EDA(df):
#    include = ['suburb', 'Street_Name', 'bed', 'bath', 'car', 'land_sqm', 'Valuation']
#    df = df[include].dropna()
    include = ['suburb', 'Beds', 'Baths', 'Car', 'Lot', 'Valuation']
    cat_list = ['suburb']
    df = df[include].dropna()
    df.Valuation = df.Valuation.astype(int)
    
    df_copy = df.copy(deep=True)

# Create DUMMYS from Categorical List
    to_drop = [n + '_' + str(df_copy[n].unique()[-1]) for n in cat_list]       # Get to_drop col names by using Categorical List

    df_copy = pd.get_dummies(df_copy, columns = cat_list, drop_first = False)  # Create dummy cols into the DataFrame

    for a in to_drop:                                                          # Double Check to_drop Columns
        if(a not in df_copy.columns.values):
            print (n)

    df_copy.drop(columns = to_drop, inplace = True)                            # Drop last dummy column
    
    column_list = df_copy.columns.tolist()
    df_copy = df_copy[column_list].dropna()
    column_list.remove('Valuation')
    print ('column List\t', column_list)
#    X = df_copy[['suburb', 'Beds', 'Baths', 'Car', 'Lot']]
    X = df_copy[column_list]
    y = df_copy['Valuation']
    return (X, y)

# Find Baseline
# def baseline (y):
#     print (y)
#     print('Baseline accuracy:', y.value_counts(normalize=True)[1]*100)
    
# Model fitting
def fit_model(X, y):
#    PREDICTOR = RandomForestClassifier(n_estimators=100).fit(X, y)
    PREDICTOR = RandomForestRegressor(n_estimators=100).fit(X, y)
    print (PREDICTOR)
    return PREDICTOR

# Serialize
def serialize(model):
    with open(pickle_file_path, 'wb') as f:
        dill.dump(model, f)

# Main Functionf
def main():
    try:
        df = load_data()
        X, y = EDA(df)
        model = fit_model(X, y)
        print (model.predict(X))
        serialize(model)
        print ('Random Forest Classifier Trained on Sellable data and Serialized')
    except Exception as err:
        print(err.args)
        exit

# Process Function
if __name__ == '__main__':
    main()