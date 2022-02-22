from re import U
from typing import Sized
from matplotlib.pyplot import title
import streamlit as st
import pandas as pd
import altair as alt
import numpy as np

# import altair as alt
# from vega_datasets import data

# source = data.population.url

# pink_blue = alt.Scale(domain=('Male', 'Female'),
#                       range=["steelblue", "salmon"])

# slider = alt.binding_range(min=1900, max=2000, step=10)
# # select_year = alt.selection_single(name="year", fields=['year'],
# #                                    bind=slider, init={'year': 2000})

# input = alt.binding_select(options=[2004,2005,2006,2007], name='Year')
# select_year = alt.selection_single(fields=['Year'], bind=input)


# pop = alt.Chart(source).mark_bar().encode(
#     x=alt.X('sex:N', title=None),
#     y=alt.Y('people:Q', scale=alt.Scale(domain=(0, 12000000))),
#     color=alt.Color('sex:N', scale=pink_blue),
#     column='age:O'
# ).properties(
#     width=20
# ).add_selection(
#     select_year
# ).transform_filter(select_year)


# st.altair_chart(pop)


#Define functions to load data
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv("health-HIV-expenditure.csv")
    return df.sort_index()

def get_indicator_slice(indicator):
    slice = df[df["Indicator Name"].isin(indicator)]
    return slice


#Loading initial data
st.title("Global Health")
with st.spinner(text="Loading data..."):
    df = load_data()
    df = pd.melt(df,id_vars=["Country Name",'Indicator Name'])
    df.rename(columns = {"variable":'Year'},inplace=True)
    # df = df.astype({'Year':'int64'})


st.dataframe(df.head(1000))

st.header("Thinking of using the example in class where the research used a slider to show the plot points move like Hans Rosling and gapminder")

#Define multi select to get indicators people can use
potentials = df["Indicator Name"].unique()
potentials = potentials.tolist()

#Remove Y variables
potentials.remove("Antiretroviral therapy coverage for PMTCT (% of pregnant women living with HIV)")
potentials.remove("Antiretroviral therapy coverage (% of people living with HIV)")

selection = st.selectbox("Select indicator(s)",options = potentials)

#If selection is made. Only input to prevent errors
if selection:
    #slice data based on indicators chosen

    #Convert string to list 
    selection = [selection]
    selection.extend(['Antiretroviral therapy coverage (% of people living with HIV)',"Antiretroviral therapy coverage for PMTCT (% of pregnant women living with HIV)"])
    # st.write(selection)
    data = get_indicator_slice(selection)
    data = data.dropna()
    # st.write(potentials)

    #Get unique countries
    unique_countries = data["Country Name"].unique()
    countries = st.multiselect("Select Countries",default=unique_countries[0],options = unique_countries)
    
    #If countries selected. Only input to prevent errors
    if countries:
        data =data[data["Country Name"].isin(countries)]

        #Pivot data and clean it up
        data = data.pivot_table(index=['Country Name',"Year"],columns="Indicator Name",values=['value'])
        data.columns = data.columns.get_level_values(1)
        data.reset_index(inplace=True)
        # data = data.dropna()
        data = data.astype({'Year':'int32'})

        st.write(data)
        select_year = alt.selection_single(name='select', fields=['Year'], 
        init={'Year': 2000},bind=alt.binding_range(min=2000, max=2020, step=1.0))


        left = alt.Chart(data,title="% of people living with HIV recieving antiretroviral therapy").mark_circle().encode(
            alt.X(selection[0],type='quantitative'),
            alt.Y(selection[1],title='Antiretroviral therapy coverage', type = "quantitative"),
            alt.Size(selection[1])
            ,alt.Color('Country Name'),
            alt.Tooltip(selection[1],type="quantitative"),
        ).add_selection(select_year).transform_filter(select_year)


        right = alt.Chart(data,title='% of pregnant women living with HIV recieving antiretroviral therapy').mark_circle().encode(
            alt.X(selection[0],type='quantitative'),
            alt.Y(selection[2],title='Antiretroviral therapy coverage', type = "quantitative"),
            alt.Size(selection[1])
            ,alt.Color('Country Name'),
            alt.Tooltip(selection[1],type="quantitative")
        ).add_selection(select_year).transform_filter(select_year)
           

        st.altair_chart(left|right)
    else:
        st.write(data.head(50))
else:
    st.write(df.head(50))





