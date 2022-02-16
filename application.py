from re import U
import streamlit as st
import pandas as pd
import altair as alt

#Define functions to load data
@st.cache
def load_data():
    df = pd.read_csv("health-raw-2021.csv",skiprows=3)
    return df.sort_index()

def get_indicator_slice(indicator):
    slice = df[df["Indicator Name"].isin(indicator)]
    return slice


st.title("Global Health")
with st.spinner(text="Loading data..."):
    df = load_data()
    df = df.loc[:,["Country Name",'Indicator Name','1960', '1961', '1962', '1963', '1964', '1965', '1966', '1967', '1968',
'1969', '1970', '1971', '1972', '1973', '1974', '1975', '1976', '1977',
'1978', '1979', '1980', '1981', '1982', '1983', '1984', '1985', '1986',
'1987', '1988', '1989', '1990', '1991', '1992', '1993', '1994', '1995',
'1996', '1997', '1998', '1999', '2000', '2001', '2002', '2003', '2004',
'2005', '2006', '2007', '2008', '2009', '2010', '2011', '2012', '2013',
'2014', '2015', '2016', '2017', '2018', '2019', '2020']]
    df = pd.melt(df,id_vars=["Country Name",'Indicator Name'])
    df.rename(columns = {"variable":'Year'},inplace=True)

st.dataframe(df.head(100))

st.header("Thinking of using the example in class where the research used a slider to show the plot points move like Hans Rosling and gapminder")

#Define multi select to get indicators people can use
potentials = df["Indicator Name"].unique()
selection = st.multiselect("Select indicator(s)",options = potentials)

#slice data based on indicators chosen
data = get_indicator_slice(selection)
# data = data.head(1500)
data = data.dropna()
st.write(data)

# select_year = alt.selection_single(
#     name='select', fields=['year'], init={'year': 1955},
#     bind=alt.binding_range(min=1960, max=2020, step=1)
# )


pop = alt.Chart(data).mark_point(filled=True).encode(
    alt.X("Year", scale=alt.Scale(zero=False)),
    alt.Y("value:Q",scale=alt.Scale(zero=False)),
    # alt.Size('value:Q', scale=alt.Scale(domain=[0, 1200000000], range=[0,1000])),
    # alt.Color('Country Name:N', legend=None),
    alt.OpacityValue(0.5),
    alt.Tooltip('Country Name:N')
    # alt.Order('pop:Q', sort='descending')
)
# .add_selection(select_year).transform_filter(select_year)

st.altair_chart(pop)
