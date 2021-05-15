import streamlit as st
import pandas as pd
import numpy as np
from module.clean_prepare import load_data



def welcome():
	#front end elements of web page
	html_page = """
	<div style ="background-color:yellow;padding:23px>
	<h1 style ="color:black;text-align:center;"> Africa Conflict Data Mining App</h1>
	"""
	#display the front end aspect 
	st.markdown(html_page, unsafe_allow_html=True)

	#title
	st.title("Conflict in Africa")

	#description
	text = """    
	Africa conflict data comes from ACLED (Armed Conflict Location & Event Data) Project 
	that report information on the type, agents, exact location, date, and other 
	characteristics of political violence events, demonstrations and select political
	relevant non-violent events.

	For more information click [here](https://acleddata.com/).
	"""

	st.markdown(text, unsafe_allow_html=True)

	#load data
	africa = load_data()


	if st.checkbox('See/Hide data', key=5):
	    st.dataframe(africa.iloc[:100, :])

	text = """
	The ACLED code: Attributes description can be taken from this
	pdf file. [codebook ACLED](https://reliefweb.int/sites/reliefweb.int/files/resources/ACLED_Codebook_2017FINAL%20%281%29.pdf).
	Data start at 1997 and end at 2020.

	## ACLED CODE: Attribute short description.

	1. **EVENT_DATE**: The day, month and year on which an event took place.
	2. **YEAR**: The year in which an event took place.
	3. **TIME_PRECISION**: A numeric code indicating the level of certainty of the date coded for the event.
	4. **EVENT_TYPE**: The type of event.
	5. **SUB_EVENT_TYPE**: The type of sub-event.
	6. **ACTOR1**: The named actor involved in the event.
	7. **INTER1**: A numeric code indicating the type of ACTOR1
	8. **ACTOR2**: The named actor involved in the event.
	9. **INTER2**: A numeric code indicating the type of ACTOR2.
	10. **INTERACTION**: A numeric code indicating the interaction between type of ACTOR1 and ACTOR2.
	11. **REGION**: The region of the world where the event took place.
	12. **COUNTRY**: The country in which the event took place.
	13. **ADMIN1**: The largest sub-national administrative region in which the event took place.
	14. **ADMIN2**: The second largest sub-national administrative region in which the event took place.
	15. **LOCATION**: The location in which the event took place.
	16. **LATITUDE**: The latitude of the location.
	17. **LONGITUDE**: The longitude of the location.
	18. **GEO_PRECISION**: A numeric code indicating the level of certainty of the location coded for the event.
	19. **SOURCE**: The source of the event report.
	20. **SOURCE SCALE**: The scale (local, regional, national, international) of the source.
	21. **NOTES**: A short description of the event.
	22. **FATALITIES**: The number of reported fatalities which occured during the event.
	"""

	with st.beta_expander("ACLED code."):
		st.markdown(text, unsafe_allow_html=True)
##*******************************************************************************************              




        



