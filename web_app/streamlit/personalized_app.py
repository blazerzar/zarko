import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import folium
from streamlit_folium import st_folium, folium_static
import time
import numpy as np  
import pandas as pd 
import plotly.express as px 
import streamlit as st  

st.set_page_config(layout="wide")

# year_list = [2019, 2020, 2021, 2022, 2023, 2024]
# with st.sidebar:
#     st.title('☀️žARKO dashboard')
#     selected_year = st.selectbox('Leto', year_list)

# st.title('Preverite svoje sončne lokacije') 
# st.write("Ta stran vam omogoča, da preverite stanje vaših sončnih celic.")

# # map = folium.Map(location=[46.0569, 14.85058], zoom_start=8.4, tiles='CartoDB positron')
# # geojson_slo = json.load(open('UE.geojson', encoding='utf-8'))
# # df = pd.read_csv("UE.csv", encoding='utf-8')
# # df["value"] = np.random.randint(1, 100, len(df))

# # choropleth = folium.Choropleth(
# #     geo_data=geojson_slo,
# #     data=df,
# #     columns=('UE_MID', 'value'),
# #     key_on='feature.properties.UE_MID',
# #     line_opacity=0.8,
# #     highlight=True,
# #     fill_color='YlOrRd',
# # )

# # for feature in choropleth.geojson.data['features']:
# #     feature['properties']['weekly'] = f"Tedensko obsevanje {np.random.randint(1, 100)} W/m2"

# # col1, col2 = st.columns([2, 1])
# # with col1:
# #     with st.container():
# #         choropleth.geojson.add_child(
# #             folium.features.GeoJsonTooltip(['UE_UIME', "weekly"], labels=False)
# #         )
# #         choropleth.geojson.add_to(map)
# #         st_map = st_folium(map, width=700, height=450, returned_objects=[])

# #     df_obs = df[['UE_UIME', 'value']]

# #     df_obs_sorted = df_obs.sort_values(by="value", ascending=False)

# with col2:
#     st.markdown('#### Sončne lokacije ')

#     st.dataframe(df_obs_sorted,
#                 column_order=("UE_UIME", "value"),
#                 hide_index=True,
#                 width=None,
#                 column_config={
#                 "UE_UIME": st.column_config.TextColumn(
#                     "UE_UIME",
#                 ),
#                 "value": st.column_config.ProgressColumn(
#                     "Obsevanje (W/m2)",
#                     format="%f",
#                     min_value=0,
#                     max_value= 150,# max(df_obs_sorted["value"]),
#                     )}
#                 )


# read csv from a github repo
dataset_url = "https://raw.githubusercontent.com/Lexie88rus/bank-marketing-analysis/master/bank.csv"

# read csv from a URL
@st.experimental_memo
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url)

df = get_data()

df_jun_daily = pd.read_csv("./jun_daily.csv")
df_hourly = pd.read_csv("./hourly_jun_3rd.csv")

# dashboard title
st.title("Preverite efektivnost vaših sončnih celic")

# year_list = [2019, 2020, 2021, 2022, 2023, 2024]
# with st.sidebar:
#     st.title('☀️žARKO dashboard')
#     selected_year = st.selectbox('Leto', year_list)

st.title('☀️ žARKO') 
st.write("dwadawdwa")

col1, col2 = st.columns([0.3, 0.7])
with col1:
    # create image on the left side
    st.image('../CM00006_202206200000_05.jpg', caption='Sunrise by the mountains')

with col2:
    # # top-level filters

    # # creating a single-element container
    placeholder = st.empty()

    # # dataframe filter
    # df = df[df["job"] == month_filter]

    df["age_new"] = df["age"] * np.random.choice(range(1, 5))
    df["balance_new"] = df["balance"] * np.random.choice(range(1, 5))

    # creating KPIs
    avg_age = np.mean(df["age_new"])

    count_married = int(
        df[(df["marital"] == "married")]["marital"].count()
        + np.random.choice(range(1, 30))
    )

    balance = np.mean(df["balance_new"])

    with placeholder.container():

        # create three columns
        kpi1, kpi2, kpi3 = st.columns(3)

        # fill in those three columns with respective metrics or KPIs
        kpi1.metric(
            label="Količina pridobljenega globalnega segrevanja 🌡️",
            value=round(avg_age),
            delta=round(avg_age) - 10,
        )
        
        st.markdown(
            """
        <style>
        [data-testid="stMetricValue"] {
            font-size: 50px;
        }
        [data-testid="stMetricLabel"] {
            font-size: 50px;
        }
        </style>
        """,
            unsafe_allow_html=True,
        )

        kpi2.metric(
            label="Količina shranjenega denarja 💰",
            value=int(count_married),
            delta=-10 + count_married,
        )
            
        choice_col, _ = st.columns([0.5, 0.5])
        with choice_col:
            pregled_choice_list = ["dnevni", "tedenski", "mesečni"]
            selected_choice = st.selectbox('Izberite pregled', pregled_choice_list)

        if selected_choice == "dnevni":
            _col1, _col2 = st.columns([0.5, 0.5])
            with _col1:
                # take every second instance
                fig2 = px.line(data_frame=df_hourly, x="hours", y="measured")
                fig2.add_scatter(x=df_hourly["hours"], y=df_hourly["predicted"])
                fig2.update_layout(yaxis_range=[0, 1200])
                st.write(fig2)
            with _col2:
                st.empty()
        elif selected_choice == "tedenski":
            _col1, _col2 = st.columns([0.5, 0.5])
            with _col1:
                # take every second instance
                fig2 = px.line(data_frame=df_hourly, x="hours", y="measured")
                fig2.add_scatter(x=df_hourly["hours"], y=df_hourly["predicted"])
                fig2.update_layout(yaxis_range=[0, 1200])
                st.write(fig2)
            with _col2:
                st.empty()
        elif selected_choice == "mesečni":
            _col1, _col2 = st.columns([0.5, 0.5])
            with _col1:
                # take every second instance
                fig2 = px.line(data_frame=df_hourly, x="hours", y="measured")
                fig2.add_scatter(x=df_hourly["hours"], y=df_hourly["predicted"])
                fig2.update_layout(yaxis_range=[0, 1200])
                st.write(fig2)
            with _col2:
                st.empty()
        

        # col1_figs, col2_figs = st.columns(2)
        
        # with col1_figs:
        #     st.markdown("### Dnevni pregled")
        #     # make plot line with plotly
        #     fig2 = px.line(data_frame=df, x="age_new", y="balance_new")
        #     # fig2 = px.histogram(data_frame=df, x="age_new")
        #     st.write(fig2)

        # with col2_figs:
        #     st.markdown("### Mesečni pregled")
        #     # make plot line with plotly
        #     fig2 = px.line(data_frame=df, x="age_new", y="balance_new")
        #     # fig2 = px.histogram(data_frame=df, x="age_new")
        #     st.write(fig2)


        # st.markdown("### Detailed Data View")
        # st.dataframe(df)
        # time.sleep(1)   