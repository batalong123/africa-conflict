import streamlit as st
from PIL import Image
from module.contact import contact_me
from module.acdm import welcome
from module.all_africa import eda, geo
from module.by_country import country
from module.prediction import predict
from module.tokenization import tokenizer


#to get connection
st.set_page_config(
page_title="Africa Conflict DM App",
page_icon= ":smiley:",
layout="centered",
initial_sidebar_state="expanded")

file = 'image/batalongCollege.png'
image = Image.open(file)
img= st.sidebar.image(image, use_column_width=True)


st.sidebar.header('****** About Africa Conflict DM ******')
st.sidebar.text(""" 
Africa Conflict DM is a Data Mining 
Board app which allows an user to
understand and discover a knownledge 
in the Africa conflict database powered
by ACLED.
Be free to enjoy this app!.
    """)

st.sidebar.title('Section')
page_name = st.sidebar.selectbox('Select page:', ("Welcome", "All Africa", "By country", "Prediction", "Contact"))


if page_name == "Welcome":
	welcome()

if page_name == 'All Africa':

	st.title('Conflict in Africa: Explore and analyse.')
	st.sidebar.title('Sub section')
	page  = st.sidebar.selectbox('Select page:', ("eda", "geospatial"))

	if page == 'eda':
		eda()

	if page == 'geospatial':	
		geo()


if page_name == "By country":
	st.title('Conflict in Africa: Explore and analyse by country.')
	country()

if page_name == "Prediction":
	st.title('Event type and region prediction.')
	text = """ 
		Suppose we have the conflict notes from a media tv or radio but
		 we do not know what type of event it is and where the event takes places? 
		 You need to make some prediction.
	"""
	st.markdown(text)
	predict()


if page_name == "Contact":
	contact_me()