import pandas as pd 
import numpy as np 
import streamlit as st


@st.cache(allow_output_mutation=True)
def load_data():
    data = pd.read_excel('data/Africa_1997-2021_Feb05.xlsx', sheet_name=0)
    data.drop(columns=['ASSOC_ACTOR_1', 'ASSOC_ACTOR_2', 'ADMIN3', 'TIMESTAMP',
                     'EVENT_ID_NO_CNTY', 'EVENT_ID_CNTY', 'ISO'], inplace=True)

    data.ACTOR2.fillna(' ', inplace=True)
    data.dropna(inplace=True)
    return data 