#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#Define functions to load data
@st.cache
def load_data():
    df = pd.read_csv("health-raw-2021.csv",skiprows=3)
    return df.sort_index()

#Loading initial data
st.title("Regression Analysis on HIV indicators from 1990 - 2020")
with st.spinner(text="Loading data..."):
    df = load_data()
    groups_df = df.groupby('Indicator Name').agg('sum')
    final_df = groups_df.transpose()
    features_df = final_df[final_df.index.isin(['1990', '1991', '1992', '1993', '1994', '1995',
'1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
'2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
'2014', '2015', '2016', '2017', '2018', '2019', '2020'])]

st.header("The original dataframe grouped by Indicators for all countries across the years")

st.dataframe(final_df.head(10))


corr_df = features_df.corr()

### Final Linear Regression Iteration ###

st.header("Pre Processing")

all_columns = ['Antiretroviral therapy coverage (% of people living with HIV)',
 'Antiretroviral therapy coverage for PMTCT (% of pregnant women living with HIV)',
 'Birth rate, crude (per 1,000 people)',
 'Births attended by skilled health staff (% of total)',
 'Cause of death, by communicable diseases and maternal, prenatal and nutrition conditions (% of total)','Cause of death, by non-communicable diseases (% of total)',
 'Community health workers (per 1,000 people)',
 'Consumption of iodized salt (% of households)',
 'Contraceptive prevalence, any methods (% of women ages 15-49)',
 'Proportion of women subjected to physical and/or sexual violence in the last 12 months (% of women age 15-49)']

selection = st.multiselect("Select indicator(s) NOT to be used",default=all_columns,options = all_columns)

final_features_df = features_df[selection + ['Adults (ages 15+) and children (ages 0-14) newly infected with HIV']]

st.write(final_features_df)
#st.dataframe(final_features_df.head(10))
# we did min max scaler / normalization

#st.header("The dataframe with relevant feature columns after normalization")
min_max_scaler = preprocessing.MinMaxScaler()
final_features_scaled_df = min_max_scaler.fit_transform(final_features_df)

final_features_scaled_df = pd.DataFrame(final_features_scaled_df, columns = min_max_scaler.get_feature_names_out())
final_features_scaled_df = final_features_scaled_df.loc[~(final_features_scaled_df['Adults (ages 15+) and children (ages 0-14) newly infected with HIV']==0)]


final_features_df = final_features_df.loc[~(final_features_df['Adults (ages 15+) and children (ages 0-14) newly infected with HIV']==0)]

#st.dataframe(final_features_scaled_df.head(10))

def regression_with_transform(final_features_df):    
    y = final_features_df['Adults (ages 15+) and children (ages 0-14) newly infected with HIV']
    
    X = final_features_df[final_features_df.columns.difference(['Adults (ages 15+) and children (ages 0-14) newly infected with HIV'])]
    
    X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size=0.4,shuffle=False)

    model_with_transform = TransformedTargetRegressor(regressor = LinearRegression(),
                                   transformer = preprocessing.MinMaxScaler(),
                                   ).fit(X_train, y_train)
    y_pred = model_with_transform.predict(X_test)
    
    return y_pred, y_test

def regression_without_transform(final_features_df):    
    y = final_features_df['Adults (ages 15+) and children (ages 0-14) newly infected with HIV']
    
    X = final_features_df[final_features_df.columns.difference(['Adults (ages 15+) and children (ages 0-14) newly infected with HIV'])]
    
    X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size=0.4,shuffle=False)

    model_without_transform = LinearRegression()
    model_without_transform.fit(X_train, y_train)
    y_pred = model_without_transform.predict(X_test)
    
    return y_pred, y_test

y_pred_without_transform,y_test_1 = regression_without_transform(final_features_scaled_df)
y_pred_with_transform,y_test= regression_with_transform(final_features_df)

#model.fit(X_train, y_train)



object_for_visualization = {'Predictions':y_pred_without_transform, "Labels": y_test_1}
viz_df = pd.DataFrame(object_for_visualization)

alt.data_transformers.enable('default', max_rows=None)

st.header("Regression Plot Predicting Adults (ages 15+) and Children (ages 0-14) Newly Infected with HIV")


chart=alt.Chart(viz_df).mark_point().encode(
    alt.X("Predictions"),
    alt.Y("Labels"),
    tooltip=["Predictions", "Labels"]

)


lin_mse = mean_squared_error(y_test_1, y_pred_without_transform)
lin_rmse = np.sqrt(lin_mse)


chart + chart.transform_regression("Predictions", "Labels").mark_line()

#st.header("Predicted adults (ages 15+) and children (ages 0-14) newly infected with HIV")

#f = min_max_scaler.inverse_transform(y_pred.reshape(-1,1))

#st.dataframe(pd.DataFrame({"Prediction":y_pred_with_transform}).head(10))



