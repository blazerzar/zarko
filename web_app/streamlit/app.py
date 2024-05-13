import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static

st.set_page_config(layout="wide")

year_list = [2019, 2020, 2021, 2022, 2023, 2024]
with st.sidebar:
    st.title('☀️žARKO dashboard')
    selected_year = st.selectbox('Leto', year_list)

st.title('☀️ žARKO') 
st.write("dwadawdwa")

map = folium.Map(location=[46.0569, 14.85058], zoom_start=8.4, tiles='CartoDB positron')
geojson_slo = json.load(open('UE.geojson', encoding='utf-8'))
df = pd.read_csv("UE.csv", encoding='utf-8')
df["value"] = np.random.randint(1, 100, len(df))

choropleth = folium.Choropleth(
    geo_data=geojson_slo,
    data=df,
    columns=('UE_MID', 'value'),
    key_on='feature.properties.UE_MID',
    line_opacity=0.8,
    highlight=True,
    fill_color='YlOrRd',
)

for feature in choropleth.geojson.data['features']:
    feature['properties']['weekly'] = f"Tedensko obsevanje {np.random.randint(1, 100)} W/m2"

col1, col2 = st.columns([2, 1])
with col1:
    with st.container():
        choropleth.geojson.add_child(
            folium.features.GeoJsonTooltip(['UE_UIME', "weekly"], labels=False)
        )
        choropleth.geojson.add_to(map)
        st_map = st_folium(map, width=700, height=450, returned_objects=[])

    df_obs = df[['UE_UIME', 'value']]

    df_obs_sorted = df_obs.sort_values(by="value", ascending=False)

with col2:
    st.markdown('#### Sončne lokacije ')

    st.dataframe(df_obs_sorted,
                column_order=("UE_UIME", "value"),
                hide_index=True,
                width=None,
                column_config={
                "UE_UIME": st.column_config.TextColumn(
                    "UE_UIME",
                ),
                "value": st.column_config.ProgressColumn(
                    "Obsevanje (W/m2)",
                    format="%f",
                    min_value=0,
                    max_value= 150,# max(df_obs_sorted["value"]),
                    )}
                )