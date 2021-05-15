
import streamlit as st 

@st.cache 
def attribute_name():
	"""
	This function gives an attributes name
	"""

	return {"obj":['EVENT_TYPE', 'SUB_EVENT_TYPE', 'ACTOR1', 'ACTOR2', 'REGION', 'COUNTRY', 'ADMIN1', 'ADMIN2', 'LOCATION', 'SOURCE', 'SOURCE_SCALE'],
	"int":['YEAR', 'TIME_PRECISION', 'INTER1', 'INTER2', 'INTERACTION', 'GEO_PRECISION', 'FATALITIES'], "float": ['LATITUDE', 'LONGITUDE']}