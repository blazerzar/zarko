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
#ge go
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title='MOJ ☀️ žARKO')


year_list = [2019, 2020, 2021, 2022, 2023, 2024]
with st.sidebar:
    st.title('Pregled')
    selected_year = st.selectbox('Leto', year_list)
    st.title('Obdobje')
    obdobje_list = ["Dnevno", "Tedensko", "Mesečno", "Letno"]
    selected_year = st.selectbox('Obdobje', obdobje_list)


# read csv from a github repo
dataset_url = "https://raw.githubusercontent.com/Lexie88rus/bank-marketing-analysis/master/bank.csv"

# read csv from a URL
def get_data() -> pd.DataFrame:
    return pd.read_csv(dataset_url)

df = get_data()

df_jun_daily = pd.read_csv("./jun_daily.csv")
df_hourly = pd.read_csv("./hourly_jun_3rd.csv")

# dashboard title
st.title("MOJ ☀️ žARKO")
st.subheader("Preverite efektivnost vaših sončnih celic")

# year_list = [2019, 2020, 2021, 2022, 2023, 2024]
# with st.sidebar:
#     st.title('☀️žARKO dashboard')
#     selected_year = st.selectbox('Leto', year_list)

#st.title('MOJ ☀️ žARKO') 

col1, col2 = st.columns([3,4])
with col1:
    # create image on the left side
    #add border to image
    #st.image('img/personal_img.jpg', caption='Zadnja slika', output_format='PNG')
    st.image('img/personal_img.jpg', caption='Zadnja slika')
    
    st.markdown(
        """
    <style>
    img {
        border: 2px solid #ddd;
        border-radius: 4px;
        padding: 5px;
        width: 100%;
    }
    </style>
    """,

        unsafe_allow_html=True,
    )

with col2:
    
    df_pred = pd.read_csv("model_output.csv")
    df_pred = df_pred.reset_index()
    df_pred["Predicted Production"] = df_pred["Predicted Production"] * 1/2.8
    df_pred["True Production"] = df_pred["True Production"] * 1/2.8
    current_time = time.strftime("%H:%M:%S", time.localtime())
    df_truncated = df_pred#[df_pred["time"] < current_time]
    fig2 = px.area(data_frame=df_truncated, x="time", y="True Production", title="Proizvodnja")
    #fig2.add_scatter(x=df_pred["time"], y=df_pred["Predicted Production"], name="Napovedana proizvodnja")
    fig2.update_layout(yaxis_range=[0, 100 + max(df_pred["True Production"].max(), df_pred["Predicted Production"].max())])
    fig2.update_yaxes(title_text='Proizvodnja')
    fig2.update_xaxes(title_text='Čas')
    fig2.update_traces(line=dict(dash='dot'))
    fig2.update_traces(fillcolor='rgba(235, 64, 52,0.9)', selector=dict(type='scatter'))
    fig2.update_traces(line=dict(color='rgba(0,0,0,1)'), selector=dict(type='scatter'))
    fig2.update_traces(line=dict(width=0), selector=dict(type='scatter'))

    fig2.add_trace(go.Scatter(x=df_pred["time"], y=df_pred["Predicted Production"],
                         line=dict(color='black', width=6, dash='dot')))

    
    predicted_sum = df_pred["Predicted Production"].sum()
    true_sum = df_pred["True Production"].sum()
    predicted_sum = int(predicted_sum)
    true_sum = int(true_sum)
    
    #add legend
    fig2.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    #add entires to legend
    
    
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
        kpi1.metric("Dnevna proizvodnja ⚡", f"{true_sum} Wh")
        kpi2.metric("Predvidena proizvodnja ☀️", f"{predicted_sum} Wh")
        kpi3.metric("Izguba 💸", f"{np.round((true_sum)/1000 * 0.16, 2)}€", delta=str(-0.25)+"€")
        #
        #make it red
        
        
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
        
    #enlarge fig2
    fig2.update_layout(width=700, height=500)
    st.write(fig2)


        # choice_col, _ = st.columns([0.5, 0.5])
        # with choice_col:
        #     pregled_choice_list = ["dnevni", "tedenski", "mesečni"]
        #     selected_choice = st.selectbox('Izberite pregled', pregled_choice_list)

        # if selected_choice == "dnevni":
        #     _col1, _col2 = st.columns([0.5, 0.5])
        #     with _col1:
        #         # take every second instance
        #         fig2 = px.line(data_frame=df_hourly, x="hours", y="measured")
        #         fig2.add_scatter(x=df_hourly["hours"], y=df_hourly["predicted"])
        #         fig2.update_layout(yaxis_range=[0, 1200])
        #         st.write(fig2)
        #     with _col2:
        #         st.empty()
        # elif selected_choice == "tedenski":
        #     _col1, _col2 = st.columns([0.5, 0.5])
        #     with _col1:
        #         # take every second instance
        #         fig2 = px.line(data_frame=df_hourly, x="hours", y="measured")
        #         fig2.add_scatter(x=df_hourly["hours"], y=df_hourly["predicted"])
        #         fig2.update_layout(yaxis_range=[0, 1200])
        #         st.write(fig2)
        #     with _col2:
        #         st.empty()
        # elif selected_choice == "mesečni":
        #     _col1, _col2 = st.columns([0.5, 0.5])
        #     with _col1:
        #         # take every second instance
        #         fig2 = px.line(data_frame=df_hourly, x="hours", y="measured")
        #         fig2.add_scatter(x=df_hourly["hours"], y=df_hourly["predicted"])
        #         fig2.update_layout(yaxis_range=[0, 1200])
        #         st.write(fig2)
        #     with _col2:
        #         st.empty()
        

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