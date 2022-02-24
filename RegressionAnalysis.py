#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
from PIL import Image
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
#from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(layout="wide",page_icon='📊')

#Define functions to load data
@st.cache
def load_data():
    df = pd.read_csv("health-raw-2021.csv",skiprows=3)
    return df.sort_index()

image = Image.open('HIV.jpg')

#Loading initial data
st.image(image)
st.title("Regression Analysis on HIV Indicators")

st.write(
    """
    
    We will be examining how certain indicators predict HIV by using a data set from 1990 to 2020.
    We will be using linear regression and ridge regression to see the HIV indicators and how strongly they predict HIV.
   .This is important as it helps us plan for future and evaluate the current strategies to mitigate HIV prevalnce.
    """)
    
with st.spinner(text="Loading data..."):
    df = load_data()
    groups_df = df.groupby('Indicator Name').agg('sum')
    final_df = groups_df.transpose()
    features_df = final_df[final_df.index.isin(['1990', '1991', '1992', '1993', '1994', '1995',
'1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
'2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
'2014', '2015', '2016', '2017', '2018', '2019', '2020'])]

st.write("**The original dataframe grouped by Indicators for all countries from 1990 - 2020**")

st.dataframe(final_df.head(10))


corr_df = features_df.corr()

### Final Linear Regression Iteration ###
st.write("**Pre Processing:**")

all_columns = ['Antiretroviral therapy coverage (% of people living with HIV)',
 'Antiretroviral therapy coverage for PMTCT (% of pregnant women living with HIV)',
 'Birth rate, crude (per 1,000 people)',
 'Births attended by skilled health staff (% of total)',
 'Cause of death, by communicable diseases and maternal, prenatal and nutrition conditions (% of total)','Cause of death, by non-communicable diseases (% of total)',
 'Community health workers (per 1,000 people)',
 'Consumption of iodized salt (% of households)',
 'Contraceptive prevalence, any methods (% of women ages 15-49)',
 'Proportion of women subjected to physical and/or sexual violence in the last 12 months (% of women age 15-49)']

selection = st.multiselect("Select indicator(s) NOT to be used in Regression:",default=all_columns,options = all_columns)

final_features_df = features_df[selection + ['Adults (ages 15+) and children (ages 0-14) newly infected with HIV']]

st.write(final_features_df)

#st.header("The dataframe with relevant feature columns after normalization")
min_max_scaler = preprocessing.MinMaxScaler()
final_features_scaled_df = min_max_scaler.fit_transform(final_features_df)

final_features_scaled_df = pd.DataFrame(final_features_scaled_df, columns = min_max_scaler.get_feature_names_out())
final_features_scaled_df = final_features_scaled_df.loc[~(final_features_scaled_df['Adults (ages 15+) and children (ages 0-14) newly infected with HIV']==0)]


final_features_df = final_features_df.loc[~(final_features_df['Adults (ages 15+) and children (ages 0-14) newly infected with HIV']==0)]
   
y = final_features_scaled_df['Adults (ages 15+) and children (ages 0-14) newly infected with HIV']

X = final_features_scaled_df[final_features_scaled_df.columns.difference(['Adults (ages 15+) and children (ages 0-14) newly infected with HIV'])]

DEFAULT = '< PICK A MODEL >'

option = st.selectbox(
     'Please choose a Regression model',
     (DEFAULT, 'Linear Regression', 'Ridge Regression'))

if option!=DEFAULT:
    st.write('You selected:', option)

st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
    }
    </style>
    """, unsafe_allow_html=True)

if option=='Linear Regression':
    
    st.header("Linear Regression:")

    print(final_features_scaled_df)
    left_column, right_column = st.columns(2)
    
    # test size
    test_size = left_column.number_input(
    				'Please choose your train-test split ratio: (range: 0.2-0.4):',
    				min_value=0.2,
    				max_value=0.4,
    				value=0.4,
    				step=0.05,
    				 )
    
    X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size=test_size,shuffle=False)
    
    model_without_transform = LinearRegression()
    model_without_transform.fit(X_train, y_train)
    y_pred = model_without_transform.predict(X_test)
        
    object_for_visualization = {'Predictions':y_pred, "Labels": y_test}
    viz_df = pd.DataFrame(object_for_visualization)
    
    alt.data_transformers.enable('default', max_rows=None)
    
    st.write("**Predicting People Newly Infected with HIV**")
    
    chart=alt.Chart(viz_df).mark_point().encode(
        alt.X("Predictions", scale=alt.Scale(zero=False)),
        alt.Y("Labels",  scale=alt.Scale(zero=False)),
        tooltip=["Predictions", "Labels"]   
    ).properties(
        width=800, height=400
    )
    chart + chart.transform_regression("Predictions", "Labels").mark_line().encode(       
        tooltip=["Predictions", "Labels"]   
        )
    
    lin_mse = mean_squared_error(y_test, y_pred)
    lin_rmse = np.sqrt(lin_mse)
    lin_rmse =round(lin_rmse, 3)
    
    
    st.write("**Mean Squared Error**")
    st.markdown('<p class="big-font">' + str(lin_rmse)+'</p>', unsafe_allow_html=True)
    
    st.write("**Feature Importance**")
    new_data = final_features_scaled_df.iloc[:, :-1]
    
    #for linear regression chart
    feature_d = {'Importance Score': model_without_transform.coef_, 'Features': list(new_data.columns)}
    feature_d= pd.DataFrame(feature_d)
    
    ### linear regresison chart   
    st.altair_chart(alt.Chart(feature_d).mark_bar().encode(        
        alt.X("Importance Score"),
        alt.Y("Features", sort = '-x'),
        tooltip=["Importance Score", "Features"]
    ).properties(
        width=800, height=400
    )
    )
    
    
elif option=='Ridge Regression':
    st.header("Ridge Regression:")
    st.write("This is a regularization technique. It is most suitable when a data set contains a higher number of predictor variables than the number of observations")
    
    print(final_features_scaled_df)
    left_column, right_column = st.columns(2)
    
    # test size
    test_size = left_column.number_input(
    				'Please choose your train-test split ratio: (range: 0.2-0.4):'
                    'α (alpha) is the parameter which balances the amount of emphasis given to minimizing RSS vs minimizing sum of square of coefficients. α can take various values: α = 0: The objective becomes same as simple linear regression',
    				min_value=0.2,
    				max_value=0.4,
    				value=0.4,
    				step=0.05,
    				 )
    
    X_train, X_test, y_train, y_test  = train_test_split(X,y, test_size=test_size,shuffle=False)
    
    st.write("**Predicting People Newly Infected with HIV**")
    left_column, right_column = st.columns(2)
    
    # test size
    alpha = left_column.number_input(
    				'Please choose Alpha for Ridge Regression (range: 0.0-1.0):',
    				min_value=0.0,
    				max_value=1.0,
    				value=0.2,
    				step=0.1,
    				 )
    
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    y_pred_ridge = ridge_model.predict(X_test)
    
    object_for_visualization = {'Predictions':y_pred_ridge, "Labels": y_test}
    ridge_viz_df = pd.DataFrame(object_for_visualization)
    
    alt.data_transformers.enable('default', max_rows=None)
    
    
    chart3=alt.Chart(ridge_viz_df).mark_point().encode(
        alt.X("Predictions", scale=alt.Scale(zero=False)),
        alt.Y("Labels",  scale=alt.Scale(zero=False)),
        tooltip=["Predictions", "Labels"]
    
    ).properties(
        width=800, height=400
    )
    chart3 + chart3.transform_regression("Predictions", "Labels").mark_line()
    
    
    lin_mse = mean_squared_error(y_test, y_pred_ridge)
    lin_rmse = np.sqrt(lin_mse)
    lin_rmse =round(lin_rmse, 3)
    
    
    
    st.write("**Mean Squared Error**")
    st.markdown('<p class="big-font">' + str(lin_rmse)+'</p>', unsafe_allow_html=True)
    
    st.write("**Feature Importance**")
    new_data = final_features_scaled_df.iloc[:, :-1]
    #st.write(list(new_data.columns))
    
    #for Ridge regression chart
    feature_d = {'Importance Score': ridge_model.coef_, 'Features': list(new_data.columns)}
    feature_d= pd.DataFrame(feature_d)
    
    ### Ridge regresison chart
    st.altair_chart(alt.Chart(feature_d).mark_bar().encode(        
        alt.X("Importance Score"),
        alt.Y("Features", sort = '-x'),
        tooltip=["Importance Score", "Features"]
    ).properties(
        width=800, height=400
    )
        )
 