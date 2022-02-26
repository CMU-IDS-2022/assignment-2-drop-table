# Aaron Ho aaronho@andrew.cmu.edu
from re import U
from typing import Sized
from matplotlib import colors
from matplotlib.pyplot import title
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

#Define functions to load data
# @st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("health-raw-2021.csv",skiprows=3)
    return df.sort_index()

def get_indicator_slice(indicator):
    slice = df[df["Indicator Name"].isin(indicator)]
    return slice


#Loading initial data
st.title("Global Health")
with st.spinner(text="Loading data..."):
    df = load_data()
    df.drop(columns=['Country Code','Indicator Code'],inplace=True)
    df = pd.melt(df,id_vars=["Country Name",'Indicator Name'])
    df.rename(columns = {"variable":'Year'},inplace=True)
    # df = df.astype({'Year':'int64'})


st.dataframe(df.head(1000))


st.write("The purpose of these charts is to show how prepared a country/region has been to HIV. The checkbox will show changes in related indicators over the years for the selected countries/regions/. Please select an indicator and the countries/regions you would like to explore")
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





