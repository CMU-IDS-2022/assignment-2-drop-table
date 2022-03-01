# Aaron Ho aaronho@andrew.cmu.edu
# Anam Iqbal anami@andrew.cmu.edu

import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
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

st.set_page_config(layout="wide",page_icon='📊')

image = Image.open('HIV.jpg')

#Loading initial data
st.image(image)
st.title("HIV Health Trends Analysis")
st.write("This application enables users to understand present HIV health trends through 1) Exploratory Data Analysis and 2) Machine Learning Models and plan for the future by evaluating strategies to mitigate HIV prevalance. Created by Aaron Ho and Anam Iqbal.")

#Define functions to load data
# @st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("health-raw-2021.csv",skiprows=3)
    return df.sort_index()

def get_indicator_slice(indicator):
    slice = df[df["Indicator Name"].isin(indicator)]
    return slice


#Loading initial data
#st.title("Global Health")
st.title("Exploratory Data Analysis")
st.write("*Note: this data reflects the metrics collected as of 2021 and may reflect missing data points from previous years.")
with st.spinner(text="Loading data..."):
    df = load_data()
    df.drop(columns=['Country Code','Indicator Code'],inplace=True)
    df = pd.melt(df,id_vars=["Country Name",'Indicator Name'])
    df.rename(columns = {"variable":'Year'},inplace=True)
    # df = df.astype({'Year':'int64'})


st.dataframe(df.head(1000))


# st.write(
#     """
#          The purpose of these charts is to show how prepared a country/region has been to HIV. 
#          The checkbox will show changes in related indicators over the years for the selected countries/regions. 
#          Please select an indicator and the countries/regions you would like to explore:
         
#     """
# )
         
#Define select to get indicators people can use
potentials = [
"Incidence of HIV, ages 15-49 (per 1,000 uninfected population ages 15-49)",
"Incidence of HIV, ages 15-24 (per 1,000 uninfected population ages 15-24)",
"Young people (ages 15-24) newly infected with HIV",
"Incidence of HIV, all (per 1,000 uninfected population)",
"Adults (ages 15+) and children (ages 0-14) newly infected with HIV",
"Children (ages 0-14) newly infected with HIV",
"Adults (ages 15-49) newly infected with HIV",
"Prevalence of HIV, male (% ages 15-24)",
"Prevalence of HIV, female (% ages 15-24)",
"Children (0-14) living with HIV",
"Prevalence of HIV, total (% of population ages 15-49)"
]

selection = st.selectbox("Select indicator(s)",options = potentials)

#If selection is made. Only input to prevent errors
if selection:
    #slice data based on indicators chosen

    #Convert string to list 
    selection = [selection]
    selection.extend(['Antiretroviral therapy coverage (% of people living with HIV)',"Antiretroviral therapy coverage for PMTCT (% of pregnant women living with HIV)","Unmet need for contraception (% of married women ages 15-49)","Contraceptive prevalence, modern methods (% of women ages 15-49)","Current health expenditure (% of GDP)","Condom use, population ages 15-24, male (% of males ages 15-24)","Condom use, population ages 15-24, female (% of females ages 15-24)"])
    data = get_indicator_slice(selection)
    data.drop(data[data["Year"] == "Unnamed: 65"].index,inplace=True)

    #Get unique countries
    unique_countries = data["Country Name"].unique()
    countries = st.multiselect("Select Countries",default=unique_countries[0],options = unique_countries)
    
    #If countries selected. Only input to prevent errors
    if countries:
        data =data[data["Country Name"].isin(countries)]

        #Store data into another dataframe to show related data 
        related_data = data.dropna()
        if not related_data.empty:
            related_data = related_data.pivot_table(index=['Country Name',"Year"],columns="Indicator Name",values=['value'])
            related_data.columns = related_data.columns.get_level_values(1)
            related_data.reset_index(inplace=True)
            related_data = related_data.astype({'Year':'int64'})
            

        # #Pivot data and clean it up
        data["value"] = data["value"].fillna(0)
        data = data.pivot_table(index=['Country Name',"Year"],columns="Indicator Name",values=['value'])
        data.columns = data.columns.get_level_values(1)
        data.reset_index(inplace=True)
        data = data.astype({'Year':'int32'})

        single_country = alt.selection_single(name="select_country", fields=["Country Name"])

        select_year = alt.selection_single(name='select', fields=['Year'], 
        init={'Year': min(data["Year"])},bind=alt.binding_range(min=min(data["Year"]), max=max(data["Year"]), step=1.0))

       
        #Visual to display when clicked
        columns1 = ["Unmet need for contraception (% of married women ages 15-49)","Contraceptive prevalence, modern methods (% of women ages 15-49)"] 
        columns2 = ["Current health expenditure (% of GDP)","Condom use, population ages 15-24, male (% of males ages 15-24)","Condom use, population ages 15-24, female (% of females ages 15-24)"]
        
        #Create related charts
        contraception = alt.Chart(related_data,title="% of married women with unmet need for contraception").mark_line().encode(
            alt.X("Year",type='temporal', scale=alt.Scale(zero=False)),
            alt.Y("Unmet need for contraception (% of married women ages 15-49)",type='quantitative',scale=alt.Scale(zero=False)),
            color=alt.condition(single_country,'Country Name' ,alt.value('lightgray')),
             ).add_selection(single_country).transform_filter(single_country)
        contraceptive_prev_modern = alt.Chart(related_data,title="% of women using modern contraceptive methods").mark_line().encode(
            alt.X("Year",type='temporal', scale=alt.Scale(zero=False)),
            alt.Y("Contraceptive prevalence, modern methods (% of women ages 15-49)",type='quantitative',scale=alt.Scale(zero=False)),
            color=alt.condition(single_country,'Country Name' ,alt.value('lightgray')),
             ).add_selection(single_country).transform_filter(single_country)
        health_expen = alt.Chart(related_data,title="Current health expenditure (% of GDP)").mark_line().encode(
            alt.X("Year",type='temporal', scale=alt.Scale(zero=False)),
            alt.Y("Current health expenditure (% of GDP)",type='quantitative',scale=alt.Scale(zero=False)),
            color=alt.condition(single_country,'Country Name' ,alt.value('lightgray')),
             ).add_selection(single_country).transform_filter(single_country)
        condom_use_male = alt.Chart(related_data,title="Current health expenditure (% of GDP)").mark_line().encode(
            alt.X("Year",type='temporal', scale=alt.Scale(zero=False)),
            alt.Y("Condom use, population ages 15-24, male (% of males ages 15-24)",type='quantitative',scale=alt.Scale(zero=False)),
            color=alt.condition(single_country,'Country Name' ,alt.value('lightgray')),
             ).add_selection(single_country).transform_filter(single_country)
        condom_use_female = alt.Chart(related_data,title="Current health expenditure (% of GDP)").mark_line().encode(
            alt.X("Year",type='temporal', scale=alt.Scale(zero=False)),
            alt.Y("Condom use, population ages 15-24, female (% of females ages 15-24)",type='quantitative',scale=alt.Scale(zero=False)),
            color=alt.condition(single_country,'Country Name' ,alt.value('lightgray')),
             ).add_selection(single_country).transform_filter(single_country)
        
           


        #Left chart
        left = alt.Chart(data,title="% of people living with HIV recieving antiretroviral therapy").mark_circle(size=100).encode(
            alt.X(selection[0],type='quantitative'),
            alt.Y(selection[1],title='Antiretroviral therapy coverage', type = "quantitative",scale=alt.Scale(domain=[0,100])),
            color=alt.condition(single_country,'Country Name' ,alt.value('lightgray')),
        ).add_selection(select_year,single_country).transform_filter(select_year)

        

        #Right Chart
        right = alt.Chart(data,title='% of pregnant women living with HIV recieving antiretroviral therapy').mark_circle(size=100).encode(
            alt.X(selection[0],type='quantitative'),
            alt.Y(selection[2],title='Antiretroviral therapy coverage', type = "quantitative",scale=alt.Scale(domain=[0,100])),
            color=alt.condition(single_country,'Country Name' ,alt.value('lightgray')),
        ).add_selection(select_year,single_country).transform_filter(select_year)
        

        st.altair_chart(left|right)

        display_relate = st.checkbox('Display related indicators by year for countries/regions selected')
        if display_relate:
                level1 = alt.hconcat(contraception, contraceptive_prev_modern)
                level2 = alt.hconcat(health_expen,condom_use_male)
                st.altair_chart(alt.vconcat(level1,level2,condom_use_female))
    else:
        st.write(data.head(50))
else:
    st.write(df.head(50))
    
    
st.title("Machine Learning Analysis")
         
# st.write(
#     """
    
#     To further our analysis, we will be examining how certain indicators predict HIV by using linear regression and ridge regression.
    
#     """)
    
with st.spinner(text="Loading data..."):
    df = load_data()
    groups_df = df.groupby('Indicator Name').agg('sum')
    final_df = groups_df.transpose()
    features_df = final_df[final_df.index.isin(['1990', '1991', '1992', '1993', '1994', '1995',
'1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
'2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
'2014', '2015', '2016', '2017', '2018', '2019', '2020'])]

st.write("**Dataframe grouped by indicators for all countries from 1990 - 2020:**")

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
    
    
    st.write("**Mean Squared Error:**")
    st.markdown('<p class="big-font">' + str(lin_rmse)+'</p>', unsafe_allow_html=True)
    
    st.write("**Feature Importance:**")
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
    st.write("This is a regularization technique. It is most suitable when a data set contains a higher number of predictor variables than the number of observations.")
    
    print(final_features_scaled_df)
    left_column, right_column = st.columns(2)
    
    # test size
    test_size = left_column.number_input(
    				'Please choose your train-test split ratio: (range: 0.2-0.4):'
                    ,
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
    				'Please choose Alpha for Ridge Regression (range: 0.0-1.0):'
        'α (alpha) is the parameter which balances the amount of emphasis given to minimizing RSS vs minimizing sum of square of coefficients. α can take various values: α = 0: The objective becomes same as simple linear regression.',
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





