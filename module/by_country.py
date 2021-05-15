import streamlit as st
import altair as alt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from scipy.stats import normaltest
import pydeck as pdk
import geopandas as gpd
from module.clean_prepare import load_data


def distribution(data=None, pays=None):
	st.header('Distribution: year 1997-2020')
	#button dist
	with st.beta_container():

		if st.button('Administrative region conflict', key=1):
			fig, ax =plt.subplots()
			data.ADMIN1.value_counts().plot(kind='bar')
			ax.set_title(f'Administrative region of {pays} where exist a conflict.')
			ax.set_ylabel('count')
			st.pyplot(fig)	
			#with st.beta_expander("Learn more"):
			#	st.markdown("""Cameroon and Middle Africa can be taken respectively as an administrative
		 	#region of Middle Africa and Africa.  """)


		if st.button('Most common Department conflict', key=2):
			fig, ax =plt.subplots()
			data.ADMIN2.value_counts()[:20].plot(kind='bar')
			ax.set_title(f'Most common Department of {pays} where exist a conflict.')
			ax.set_ylabel('count')
			st.pyplot(fig)
			with st.beta_expander("Learn more"):
				st.markdown(""" We take only 20 most commons Department.  """)

		if st.button('Most common location conflict', key=4):
			fig, ax =plt.subplots()
			data.LOCATION.value_counts()[:20].plot(kind='bar')
			ax.set_ylabel('count')
			ax.set_title('Most common location where occurs a conflict.')
			st.pyplot(fig)
			with st.beta_expander("Learn more"):
				st.markdown(""" We take only 20 most commons Location.  """)

		if st.button('Most common source media', key=5):
			fig, ax =plt.subplots()
			data.SOURCE.value_counts()[:20].plot(kind='bar')
			ax.set_ylabel('count')
			ax.set_title('Most common different sources media.')
			st.pyplot(fig)
			with st.beta_expander("Learn more"):
				st.markdown(""" We take only 20 most commons sources media.  """)


		if st.button('Most common source scale', key=6):
			fig, ax =plt.subplots()
			data.SOURCE_SCALE.value_counts()[:20].plot(kind='bar')
			ax.set_title('Source scale')
			ax.set_ylabel('count')
			st.pyplot(fig)
			with st.beta_expander("Learn more"):
				st.markdown(""" We take only 20 most commons source scale.  """)

		if st.button("Conflict event type", key=7):
			fig, ax = plt.subplots()
			data.EVENT_TYPE.value_counts().plot(kind='bar')
			ax.set_title('The event type of conflict in Cameroon.')
			ax.set_ylabel('count')
			st.pyplot(fig)

	if st.checkbox('Conflict sub event type', key=1):
		st.subheader('Conflict sub event.')
		sub_event = dict()
		for u in data.EVENT_TYPE.unique():
		    sub_event[u] = data[data.EVENT_TYPE == u].SUB_EVENT_TYPE.value_counts()

		u = st.selectbox('Select event type:' , tuple(sub_event.keys()))

		figure, ax = plt.subplots() 
		sub_event[u].plot(kind='bar', ax=ax)
		ax.set_ylabel('counts')
		ax.set_title(f'Event conflict type: {u}.')
		st.pyplot(figure)
		with st.beta_expander("Learn more"):
			st.markdown("""Along the x-axis we have a sub event type, y-axis a counts.""")


	if st.button('Interaction', key=9):
		fig, ax = plt.subplots()
		data.INTERACTION.value_counts()[:20].plot(kind='bar')
		ax.set_title('Interacton distribution')
		ax.set_ylabel('counts')
		st.pyplot(fig)
		with st.beta_expander("Learn more"):
			st.markdown(""" For more detail click 
			[codebook ACLED](https://reliefweb.int/sites/reliefweb.int/files/resources/ACLED_Codebook_2017FINAL%20%281%29.pdf)
			  """)

def desc(data=None, pays=None):
	#discriptive analysis
	st.header(f"Descriptive analysis: {pays}.")
	if st.checkbox('Descriptive analysis', key=3):
		if st.button('Describe data', key=14):
			st.subheader('Describe data')
			st.dataframe(data.describe())

		
		if st.checkbox('correlation', key=4):
			st.subheader('Correlation')
			st.dataframe(data.corr())
			with st.beta_expander('Learn more'):
				st.markdown(""" 
				Two variables are positive correlated if only if

				1. $corr >= 0.5$.
				2. $corr >= 0.8$ (strong).

				Two variables are negative correlated if only if

				1. $corr < -0.5$.
				2. $corr <= -0.8$ (strong).

				Assume that $x$ and $y$ are two independent variables; if we compute $corr(x,y) >= 0.5$ or $corr(x,y) < -0.5$ or 
				$corr(x,y) >= 0.8$ or $corr(x,y)<= -0.8$,
				we can plot this
				> $y = f(x)$ this means that the trend of $y$ depends on trend of $x$. 

				**N.B**: 

				1. Positive correlation means that $x$ and $y$ go together i.e if $x$ increase over time, $y$ increase over time.
				2. Negative correlation means that $x$ and $y$ does not go together if $x$ increase over time, $y$ decrease over time.

				""")

			year = st.selectbox('Select year:', range(1997,2021))

			#correlation for each year
			result = 'corr(latitude, longitude) = {}.'.format(data[data.YEAR==year].corr().loc['LATITUDE', ['LONGITUDE']].values[0])
			st.success(result)

			fig, ax = plt.subplots()
			sns.regplot(x='LONGITUDE', y='LATITUDE', data=data[data.YEAR==year], lowess=True)
			title = 'Regression plot latitude and longitude for year {}'.format(year)
			ax.set_title(title)
			st.pyplot(fig)


			cols = ['EVENT_TYPE','ADMIN1','LOCATION','SUB_EVENT_TYPE','FATALITIES', 'EVENT_DATE']
			if st.checkbox('Geolocalization',key=5):
				st.subheader('Geolocalization')
				st.dataframe(data[data.YEAR==year][cols])
				st.dataframe(data[data.YEAR==year][['EVENT_DATE','LOCATION','NOTES', 'FATALITIES']])

			@st.cache
			def curiosity():

				#initialize
				corr = []
				years = []
				total_fata = []
				admin1 = []
				ev_tpe = []
				sub_type = []

				for u in range(1997,2021):

					corr.append(data[data.YEAR==u].corr().loc['LATITUDE', ['LONGITUDE']].values[0])
					years.append(u)
					admin1.append(data[data.YEAR==u].ADMIN1.mode().values)
					total_fata.append(data[data.YEAR==u].FATALITIES.sum())
					ev_tpe.append(data[data.YEAR==u].EVENT_TYPE.mode().values)
					sub_type.append(data[data.YEAR==u].SUB_EVENT_TYPE.mode().values)

				cdata = pd.DataFrame()

				cdata['corr(lat,long)'] = corr
				cdata['year'] = years
				cdata['total_fatalities'] = total_fata
				cdata['admin1_mode'] = admin1
				cdata['event_type_mode'] = ev_tpe
				cdata['sub_event_type_mode'] = sub_type

				return cdata

			if st.checkbox('Some curiosity',key=6):
				st.subheader('Relevant informative data')
				df = curiosity()
				cd = df.set_index('year')
				st.dataframe(df)

				if st.button('plot corr(lat,long) vs total_fatalities'):
					c = alt.Chart(df).mark_bar().encode(x='corr(lat,long)', y='total_fatalities', 
						tooltip=['corr(lat,long)', 'total_fatalities'])
					st.altair_chart(c, use_container_width=True)

				if st.button('Heatmap calendar'):
					
					fig, ax = plt.subplots()
					sns.heatmap(cd[['corr(lat,long)', 'total_fatalities']],center=0, annot=True, fmt='.6g')
					ax.set_title('Heatmap calendar.')
					st.pyplot(fig)

		#conflict is spreading
		if st.checkbox('is conflict spreading?', key=7):
			st.subheader('Conflict is spreading.')
			year_fata = data[data.YEAR!=2021] .groupby('YEAR')['FATALITIES'].agg('sum')
			#event type section
			if st.checkbox('event type', key=8):
				st.subheader('Event type')
				if st.button('fatalities barplot'):
					fig, ax = plt.subplots()
					year_fata.plot(kind='bar')
					ax.set_ylabel('cummulative fatalities')
					ax.set_title(F'Progresssive of fatalities caused by conflict in {pays}.')
					st.pyplot(fig)

				event_conflict = pd.pivot_table(data, values='FATALITIES', 
						columns='EVENT_TYPE', index='YEAR', aggfunc='sum')

				if st.button('calendar event type', key=15):
					fig, ax = plt.subplots()
					sns.heatmap(event_conflict, center=0, annot=True, fmt='.6g')
					ax.set_title(F'Heatmap of conflict in {pays}.')
					st.pyplot(fig)

					with st.beta_expander('Learn more'):
						st.markdown("""
						The blank space means that no data are recorded in that year corresponding to the event type. 
						 """)

				if st.button('event type describe', key=16):
					st.dataframe(event_conflict.describe())

				if st.button('event type similarity', key=17):
					st.dataframe(event_conflict.corr())
					with st.beta_expander('Learn more'):
						st.markdown("""
						Refer to correlation learn more section.
						 """)

				if st.button('sub event similarity', key=18):
					sub_conflict = pd.pivot_table(data, values='FATALITIES', index='YEAR',
					 columns='SUB_EVENT_TYPE', aggfunc='sum')

					st.dataframe(sub_conflict.corr())
					with st.beta_expander('Learn more'):
						st.markdown("""
						NaN: Not a Number.  
						 """)

			# Administrative region
			if st.checkbox('conflict administrative region', key=9):
				st.subheader('Conflict administrative region')
				region = pd.pivot_table(data, values='FATALITIES', columns='ADMIN1',
				 index='YEAR', aggfunc='sum')

				if st.checkbox('fatalities calendar'):
					if st.button('scaling'):
						fig, ax = plt.subplots(figsize=(15, 5), dpi=150)
						fmt = '.2g'
						annot = False
					else:
						fig, ax = plt.subplots()
						fmt = '.3g'
						annot = True
					sns.heatmap(region, annot=annot, fmt=fmt)
					ax.set_title(f'Fatalities calandar conflict in {pays} ')
					st.pyplot(fig)

				if st.button('conflict describe'):
					st.dataframe(region.describe())

				if st.button('conflict similarity'):
					st.dataframe(region.corr())

					with st.beta_expander('Learn more'):
						st.markdown("""
						correlation give similarity between two variables for data going to 1997 to 2020.
						  """)

def geo(data=None, pays=None):

	st.header(f'Geospatial analysis: {pays}')
	year = st.selectbox('Select year conflict:', range(1997, 2021))
	year_data = data[data.YEAR == year]

	st.subheader('Map')
	if st.checkbox('Map'):

		st.map(pd.DataFrame(year_data[['LATITUDE', 'LONGITUDE']].values, columns=['lat', 'lon']), zoom=6)


		if st.checkbox('Admin barh'):
	    		
			st.subheader('Fatalities by admin1.')

			fig, ax = plt.subplots()

			ax.barh(year_data.ADMIN1, year_data.FATALITIES)
			ax.set_xlabel('Fatalities')
			ax.set_ylabel('Administrative')
			ax.set_title(f'{year} Geospatial fatalities.')
			st.pyplot(fig)


	st.subheader('Relation between attributes.')
	if st.checkbox('Attributes relation.'):

		st.write('### Relation between INTER1 (ACTOR1) and INTER2 (ACTOR2).')
		if st.checkbox('ACTOR1 vs ACTOR2.'):
			
			interaction = pd.pivot_table(year_data, values='INTERACTION', index='INTER2',
			 columns='INTER1', aggfunc=np.count_nonzero)
			fatalities_interaction =  pd.pivot_table(year_data, values='FATALITIES',
			 index='INTER2', columns='INTER1', aggfunc=np.sum)

			with st.beta_expander('Learn more'):
				text = """

				Please use this pdf file [codebook ACLED](https://reliefweb.int/sites/reliefweb.int/files/resources/ACLED_Codebook_2017FINAL%20%281%29.pdf).
				section Interaction.
				"""
				st.markdown(text, unsafe_allow_html=True)

			with st.beta_container():

				fig, ax = plt.subplots(figsize=(12,6))
				sns.heatmap(interaction, center=0, annot=True, fmt='.5g', ax=ax)
				ax.set_title(f'{pays}: Number of conflict INTER1 vs INTER2 for year {year}.')
				ax.set_xlabel('INTER1 (ACTOR1)')
				ax.set_ylabel('INTER2 (ACTOR2)')
				st.pyplot(fig)

				fig2, ax2 = plt.subplots(figsize=(12,6))
				sns.heatmap(fatalities_interaction, center=0, annot=True, fmt='.6g',ax=ax2)
				ax2.set_title(f'{pays}: Fatalities caused by conflict ACTOR1 vs ACTOR2 for year {year}.')
				ax2.set_xlabel('INTER1 (ACTOR1)')
				ax2.set_ylabel('INTER2 (ACTOR2)')
				st.pyplot(fig2)


		st.write(f'### In each admin1, how many fatalities the actors make in the conflict for year {year}?')
		if st.checkbox('FATALITIES caused by ACTOR1/ADMIN1.'):

			region_actor = pd.pivot_table(year_data, values='FATALITIES', index='INTER1',
			 columns='ADMIN1', aggfunc=np.sum)

			with st.beta_container():
				fig, ax = plt.subplots(figsize=(12, 6))
				sns.heatmap(region_actor, center=0, annot=True, fmt='.6g')
				ax.set_title(f'{pays}: Fatalities in each admin1 caused by ACTOR1 for {year}.')
				ax.set_ylabel('ACTOR1')
				st.pyplot(fig)

				st.write(f'### {pays}: Total fatalities caused by ACTOR1 for {year}.')
				st.bar_chart(region_actor.sum(axis=1))

		st.write(f'### In each type of conflict event, how many fatalities the actors make for year {year}?')
		if st.checkbox('FATALITIES caused by ACTOR1/EVENT_TYPE.'):

			event_actor = pd.pivot_table(year_data, values='FATALITIES', index='INTER1',
			 columns='EVENT_TYPE', aggfunc=np.sum)

			with st.beta_container():

				fig, ax = plt.subplots(figsize=(12,6))
				sns.heatmap(event_actor, center=0, annot=True, fmt='.6g', ax=ax)
				ax.set_title(f'{pays}: Fatalities in each event_type caused by ACTOR1 for year {year}.')
				ax.set_ylabel('ACTOR1')
				st.pyplot(fig)

				st.write(f'### {pays}: Total fatalities in each EVENT_TYPE caused by ACTOR1.')
				st.bar_chart(event_actor.sum())


		st.write(f'### In each admin1, how many fatalities are there in each type of conflict event for year {year}?')
		if st.checkbox('FATALITIES in each ADMIN1/EVENT_TYPE.'):

			event_region = pd.pivot_table(year_data, values='FATALITIES', index='ADMIN1',
			 columns='EVENT_TYPE', aggfunc=np.sum)

			with st.beta_container():
				fig, ax = plt.subplots(figsize=(12,6))
				sns.heatmap(event_region, center=0, annot=True, fmt='.6g', ax=ax)
				ax.set_title(f'{pays}: Fatalities in each admin1 by event_type.')
				st.pyplot(fig)

				st.write(f'### {pays}: Total fatalities per admin1.')
				st.bar_chart(event_region.sum(axis=1))


def country():

	data = load_data()

	ctry = st.sidebar.selectbox('Select country:', list(data['COUNTRY'].unique()))

	country_data = data[data.COUNTRY == ctry]

	page = st.selectbox('Select page:', ['Distribution viz', 'Descriptive analysis', 'Geospatial analysis'])

	if page == 'Distribution viz':
		distribution(data=country_data, pays=ctry)

	if page == 'Descriptive analysis':
		desc(data=country_data, pays=ctry)

	if page == 'Geospatial analysis':
		geo(data=country_data, pays=ctry)



