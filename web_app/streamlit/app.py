import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import folium
from streamlit_folium import st_folium
from streamlit_folium import folium_static

IMAGES_PREDS = {
    "Ljubljana" : 1081,
    "Maribor" : 916,
    "Novo mesto" : 741
}

st.set_page_config(layout="wide")

year_list = [2019, 2020, 2021, 2022, 2023, 2024]
with st.sidebar:
    st.title('Pregled')
    selected_year = st.selectbox('Leto', year_list)
    st.title('Obdobje')
    obdobje_list = ["Tedensko", "Mesečno", "Letno"]
    selected_year = st.selectbox('Obdobje', obdobje_list)

st.title('☀️ žARKO') 

map = folium.Map(location=[46.0569, 14.85058], zoom_start=8.4, tiles='CartoDB positron')
geojson_slo = json.load(open('UE.geojson', encoding='utf-8'))
df = pd.read_csv("UE.csv", encoding='utf-8')
df["value"] = np.arange(1, len(df) + 1)
df["value"] = df["ENRG"]

IMAGES_PREDS = {}

np.random.seed(42)
for uime in df["UE_UIME"]:
    #what row has the same name as uime
    row = df.loc[df["UE_UIME"] == uime]
    IMAGES_PREDS[uime] = row["value"].values[0] + np.random.randint(-100, 100)


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
    value = df.loc[df["UE_UIME"] == feature['properties']['UE_UIME']]["value"].values[0]
    feature['properties']['weekly'] = f"Tedensko obsevanje {value} W/m2"
    print(df.loc[df["UE_UIME"] == feature['properties']['UE_UIME']]["value"].values[0])

col1, col2 = st.columns([2, 1])



with col1:
    choropleth.geojson.add_child(
        folium.features.GeoJsonTooltip(['UE_UIME', "weekly"], labels=False)
    )
    choropleth.geojson.add_to(map)
    #Add marker (click on map an alert will display with latlng values)

    st_map = st_folium(map, width=700, height=450)

    loc_name = ''
    if st_map['last_active_drawing']:
        loc_name = st_map['last_active_drawing']['properties']['UE_UIME']

    #st.write(loc_name)

    if loc_name in IMAGES_PREDS:
        st.image(f"img/{loc_name}.jpg", caption=loc_name, width=700)    



with col2:
    st.markdown('#### Sončne lokacije ')
    df_obs = df[['UE_UIME', 'value']]
    df_obs_sorted = df_obs.sort_values(by="value", ascending=False)

    df_copy = df_obs_sorted.copy()

    edited_df = st.data_editor(
        df_copy,
        hide_index=True,
        column_config={
                "UE_UIME": st.column_config.TextColumn(
                    "UE_UIME",
                ),
                "value": st.column_config.ProgressColumn(
                    "Obsevanje (W/m2)",
                    format="%f",
                    min_value=0,
                    max_value= 1500,# max(df_obs_sorted["value"]),
                    ),
                },
        disabled=df.columns,
    )
    if loc_name in IMAGES_PREDS:
        st.markdown(f"### Predvideno obsevanje:")

        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = IMAGES_PREDS[loc_name],
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': f"Obsevanje W/m2 <br> {loc_name}", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [None, 1200], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#fd8d3c"},
        }))
        fig.update_layout(width=300, height=300)

        st.plotly_chart(fig, )



