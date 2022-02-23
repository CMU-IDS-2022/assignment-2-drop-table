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
# eswatini = df[df["Country Name"]=="Eswatini"]
# st.write(eswatini)


st.header("Thinking of using the example in class where the research used a slider to show the plot points move like Hans Rosling and gapminder")

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
        st.write(data)

        # #Pivot data and clean it up
        data["value"] = data["value"].fillna(0)
        data = data.pivot_table(index=['Country Name',"Year"],columns="Indicator Name",values=['value'])
        # st.write(data)
        data.columns = data.columns.get_level_values(1)
        data.reset_index(inplace=True)
        # data = data.dropna()
        data = data.astype({'Year':'int32'})

        # st.write(data)
        select_year = alt.selection_single(name='select', fields=['Year'], 
        init={'Year': min(data["Year"])},bind=alt.binding_range(min=min(data["Year"]), max=max(data["Year"]), step=1.0))

        #Allow users to select on a point to display something
        single_country = alt.selection_single(name="select_country", fields=["Country Name"])
        single_year = alt.selection_single(name="select_year",fields=["Year"])
        
        
        #Left chart
        left = alt.Chart(data,title="% of people living with HIV recieving antiretroviral therapy").mark_circle().encode(
            alt.X(selection[0],type='quantitative'),
            alt.Y(selection[1],title='Antiretroviral therapy coverage', type = "quantitative"),
            alt.Size(selection[1]),
            color=alt.condition(single_country,'Country Name' ,alt.value('lightgray'))
        ).add_selection(select_year,single_country,single_year).transform_filter(select_year)

        #Right Chart
        right = alt.Chart(data,title='% of pregnant women living with HIV recieving antiretroviral therapy').mark_circle().encode(
            alt.X(selection[0],type='quantitative'),
            alt.Y(selection[2],title='Antiretroviral therapy coverage', type = "quantitative"),
            alt.Size(selection[1]),
            color=alt.condition(single_country,'Country Name' ,alt.value('lightgray')),
        ).add_selection(select_year,single_country,single_year).transform_filter(select_year)
        

        st.altair_chart(left|right)

        # st.write(single_year)
        #Visual to display when clicked
        #Repeating graphs to show related metrics to the indicator chosen
        columns = ["Unmet need for contraception (% of married women ages 15-49)","Contraceptive prevalence, modern methods (% of women ages 15-49)","Current health expenditure (% of GDP)","Condom use, population ages 15-24, male (% of males ages 15-24)","Condom use, population ages 15-24, female (% of females ages 15-24)"] 
        # rows = ["Country Name"]
        #Add related repeating charts
        related = alt.Chart(data).mark_bar().encode(
                alt.X(alt.repeat("row"),type='quantitative'),
                alt.Y(alt.repeat("column"),type='quantitative'),
               alt.Color('Country Name'),
               alt.Tooltip("Year")
            ).repeat(
                column=columns,
                row=[selection[0]]
            ).add_selection(single_country,single_year).transform_filter(single_country).transform_filter(single_year)


# .transform_filter(single_country).transform_filter(single_year)
        st.altair_chart(related)
    
    else:
        st.write(data.head(50))
else:
    st.write(df.head(50))





