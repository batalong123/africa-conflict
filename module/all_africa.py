import streamlit as st 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import seaborn as sns 
from module.clean_prepare import load_data
from module.meta_data import attribute_name 
from statsmodels.graphics.mosaicplot import mosaic
from module.tokenization import word_tokenizer
import pydeck as pdk
import spacy
import spacy_streamlit as spat 
from wordcloud import WordCloud
import nltk
from spacy.matcher import PhraseMatcher
import calplot as cplt

def eda():
	"""
	EDA: Exploratory data analysis
	"""
	data = load_data()
	cols = attribute_name()
	st.sidebar.title('EDA')
	page = st.sidebar.selectbox('select page:',("Analyse", "Visualize"))

	if page == "Analyse":


		st.write('## Descriptive statistics.')
		if st.checkbox('Summary statistic.'):

			with st.beta_container():

				st.write("### Describe.")
				st.dataframe(data.describe())

				st.write("### Mode")
				st.dataframe(data.mode())

				st.write("### Skew.")
				st.dataframe(data.skew())

				st.write("### Kurtosis")
				st.dataframe(data.kurtosis())

				st.write("### Correlation.")
				st.dataframe(data.corr())

		with st.beta_expander('Learn more.'):
			text = """
				**Descriptive statistics** summarizes or describe the characteristics of a data set.

				1. **Mode** is the value that appears most often in a set of data values.
				2. **Skew** is a measure of the asymmetry of the probability distribution of a real-valued random variable about its mean.
				3. **Kurtosis** is a measure of the tailedness of the probability distribution of a real-valued random variable.
				Kurtosis measure also outlier.
				4. **Correlation** or **dependence** is any statistical relationship, whether causal or not, between two random variable data. 
			"""
			st.markdown(text, unsafe_allow_html=False)


	if page == "Visualize":
		st.write("## Exploration and Visualization.")

		if st.checkbox('Numeric attributes.'):
			st.write("### Numeric attribute.")
			with st.beta_expander("Learn more"):
				text = """
					Do not forget to correspond descriptive statistics with visualization.
					#### Precision on INTER1, INTER2 and INTERACTION attributes

					1. 0 -> *Empty*
					2. 1 -> *State Forces*
					3. 2 -> *Rebel Group*
					4. 3 -> *Political Militias*
					5. 4 -> *Identity Militias*
					6. 5 -> *Rioters*
					7. 6 -> *Protesters*
					8. 7 -> *Civilians*
					9. 8 -> *External/Other Forces*

					Where INTER1 = {1,2,3,4,5,6,7,8} and INTER2 = {0,1,2,3,4,5,6,7,8}.
					 If we have INTER1 == 1 and INTER2 == 0, we get INTERACTION == 10. 

					 **NB**: INTER1 is the code of  ACTOR1  and INTER2 is the code of ACTOR2.

					**Detail pdf file**: [codebook ACLED](https://reliefweb.int/sites/reliefweb.int/files/resources/ACLED_Codebook_2017FINAL%20%281%29.pdf).
				"""
				st.markdown(text, unsafe_allow_html=True)

			if st.checkbox('Histogram'):
				
				fig = plt.figure(figsize=(20,10))
				fig.subplots_adjust(wspace=0.3, hspace=0.5)
				for i, u in enumerate(cols['int']+cols['float']):
					ax = fig.add_subplot(3,3,i+1)
					sns.histplot(x=u, bins=20, data=data)
					ax.set_title('1997-2020 Histogram: ' +u)
				st.pyplot(fig)

			if st.checkbox('Boxplot'):

				fig = plt.figure(figsize=(20,10))
				fig.subplots_adjust(wspace=0.3, hspace=0.5)
				for i, u in enumerate(cols['int']+cols['float']):
					ax = fig.add_subplot(3,3,i+1)
					sns.boxplot(x=u, data=data)
					ax.set_title('1997-2020 Boxplot: ' +u)
				st.pyplot(fig)

		if st.checkbox('Categorical attributes.'):

			st.write('### Categorical attributes.')

			if st.checkbox('Pie'):

				st.write('#### Pie plot.')
				fig = plt.figure(figsize=(15,10))
				fig.subplots_adjust(wspace=0.2)

				for i , u in enumerate(['EVENT_TYPE', 'REGION']):
					ax = fig.add_subplot(1, 2, i+1)
					df = data[u].value_counts()
					ax.pie(df, labels=df.index, startangle=90, shadow=True)
					ax.legend(title=f'1997-2020 {u}:', loc='upper left')
				st.pyplot(fig)


			if st.checkbox('Mosaic'):

				st.write('#### Mosaic plot.')
				with st.beta_container():

					for u in ['SUB_EVENT_TYPE', 'COUNTRY', 'SOURCE_SCALE']:
						fig, ax = plt.subplots(figsize=(5,5))
						mosaic(data[u].value_counts(), horizontal=False, ax=ax, axes_label=False, title=f'1997-2020 Mosaic: {u}.')
						st.pyplot(fig)

			if st.checkbox("Barh"):

				st.write('#### Bar horizontal plot.')
				for v in [['ACTOR1', 'ACTOR2'], ['ADMIN1', 'ADMIN2'], ['LOCATION', 'SOURCE']]:
					with st.beta_container():

				
						fig  = plt.figure(figsize=(20,15
							), )
						fig.subplots_adjust(wspace=0.2, hspace=0.2)

						for i, u in enumerate(v):
							ax = fig.add_subplot(3, 2, i+1)
							df = data[u].value_counts()[:10]
							ax.barh(df.index, df.values)
							ax.set_title(f'1997-2020 {u}: 10 most commons.')
		
						st.pyplot(fig)


		if st.checkbox('Attributes relation.'):

			st.write('### Relation between attributes.')

			if st.checkbox('ACTOR1 vs ACTOR2.'):
				st.write('#### Relation between INTER1 (ACTOR1) and INTER2 (ACTOR2).')

				interaction = pd.pivot_table(data, values='INTERACTION', index='INTER2',
				 columns='INTER1', aggfunc=np.count_nonzero)
				fatalities_interaction =  pd.pivot_table(data, values='FATALITIES',
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
					ax.set_title('Number of conflict INTER1 vs INTER2 between 1997 and 2020.')
					ax.set_xlabel('INTER1 (ACTOR1)')
					ax.set_ylabel('INTER2 (ACTOR2)')
					st.pyplot(fig)

					fig1, ax1 = plt.subplots(figsize=(12,6))
					mask1 = np.triu(np.ones_like(interaction.corr()))
					sns.heatmap(interaction.corr(), center=0, annot=True, fmt='0.3g', mask=mask1, ax=ax1)
					ax1.set_title('Similarity conflict ACTOR1 vs ACTOR2 between 1997 and 2020.')
					ax1.set_xlabel('INTER1 (ACTOR1)')
					ax1.set_ylabel('INTER2 (ACTOR2)')
					st.pyplot(fig1)

					fig2, ax2 = plt.subplots(figsize=(12,6))
					sns.heatmap(fatalities_interaction, center=0, annot=True, fmt='.6g',ax=ax2)
					ax2.set_title('Fatalities caused by conflict ACTOR1 vs ACTOR2 between 1997 and 2020.')
					ax2.set_xlabel('INTER1 (ACTOR1)')
					ax2.set_ylabel('INTER2 (ACTOR2)')
					st.pyplot(fig2)

					fig3, ax3 = plt.subplots(figsize=(12,6))
					mask2 = np.triu(np.ones_like(fatalities_interaction.corr()))
					sns.heatmap(fatalities_interaction.corr(), center=0, annot=True,
					 fmt='0.3g', mask=mask2)
					ax3.set_title('Similarity fatalities ACTOR1 vs ACTOR2 between 1997 and 2020.')
					ax3.set_xlabel('INTER1 (ACTOR1)')
					ax3.set_ylabel('INTER2 (ACTOR2)')
					st.pyplot(fig3)


			if st.checkbox('FATALITIES caused by ACTOR1/REGION.'):
				st.write('#### In each region, how many fatalities the actors make in the conflict between 1997 and 2020?')

				region_actor = pd.pivot_table(data, values='FATALITIES', index='INTER1',
				 columns='REGION', aggfunc=np.sum)

				with st.beta_container():
					fig, ax = plt.subplots(figsize=(12, 6))
					sns.heatmap(region_actor, center=0, annot=True, fmt='.6g')
					ax.set_title('Fatalities in each region caused by ACTOR1 between 1997 and 2020.')
					ax.set_ylabel('ACTOR1')
					st.pyplot(fig)

					st.write('#### Total fatalities caused by ACTOR1 between 1997 and 2020.')
					st.bar_chart(region_actor.sum(axis=1))

					fig1, ax1 = plt.subplots(figsize=(12,6))
					mask = np.triu(np.ones_like(region_actor.corr()))
					sns.heatmap(region_actor.corr(), center=0, annot=True, fmt='0.3g', mask=mask)
					ax1.set_title('Similarity fatalities caused by actor1 in each region of Africa.')
					st.pyplot(fig1)

			if st.checkbox('FATALITIES caused by ACTOR1/EVENT_TYPE.'):
				st.write('#### In each type of conflict event, how many fatalities the actors make between 1997 to 2020?')

				event_actor = pd.pivot_table(data, values='FATALITIES', index='INTER1',
				 columns='EVENT_TYPE', aggfunc=np.sum)

				with st.beta_container():

					fig, ax = plt.subplots(figsize=(12,6))
					sns.heatmap(event_actor, center=0, annot=True, fmt='.6g', ax=ax)
					ax.set_title('Fatalities in each event_type caused by ACTOR1 between 1997 to 2020.')
					ax.set_ylabel('ACTOR1')
					st.pyplot(fig)

					st.write('#### Total fatalities in each EVENT_TYPE caused by ACTOR1 between 1997 and 2020.')
					st.bar_chart(event_actor.sum())

					fig1, ax1 = plt.subplots(figsize=(12,6))
					mask3 = np.triu(np.ones_like(event_actor.corr()))
					sns.heatmap(event_actor.corr(), center=0, annot=True, fmt='0.3g', mask=mask3)
					ax.set_title('Similarity fatalities caused by actor1 in different event_type.')
					st.pyplot(fig1)


			if st.checkbox('FATALITIES in each REGION/EVENT_TYPE.'):
				st.write('#### In each region, how many fatalities are there in each type of conflict event between 1997 to 2020?')

				event_region = pd.pivot_table(data, values='FATALITIES', index='REGION',
				 columns='EVENT_TYPE', aggfunc=np.sum)

				with st.beta_container():
					fig, ax = plt.subplots(figsize=(12,6))
					sns.heatmap(event_region, center=0, annot=True, fmt='.6g', ax=ax)
					ax.set_title('Fatalities in each region by event_type between 1997 to 2020.')
					st.pyplot(fig)

					st.write('#### Total fatalities per region between 1997 and 2020.')
					st.bar_chart(event_region.sum(axis=1))


def geo():
	"""
	Geospatial analysis
	"""

	data = load_data()
	cols = attribute_name()
	need_cols = ["ACTOR1", "ACTOR2", "REGION", "COUNTRY", "ADMIN1", "ADMIN2",
	"LOCATION", "EVENT_TYPE", "SUB_EVENT_TYPE", "FATALITIES", "YEAR", "LONGITUDE","LATITUDE"]

	nlp = spacy.load('en_core_web_sm')

	ext_data = data[need_cols]

	def cloud_plot(df):
		fig, ax = plt.subplots(figsize=(5,5))
		cloud = WordCloud().generate(df)
		ax.imshow(cloud, interpolation="bilinear")
		ax.axis('off')
		st.pyplot(fig)

	st.sidebar.title('Geospatial')
	page = st.sidebar.selectbox('select page:', ('Global views', 'Region views'))
	st.write('## Geospatial analysis.')

	if page == 'Global views':

		with  st.beta_container():
			st.write('### Conflict by year.')

			year = st.selectbox('Select year:', range(1997, 2021))
			year_data = ext_data[ext_data['YEAR'] == year]


			st.write(f"#### Conflict in Africa for year: {year}.")
			viewport = pdk.ViewState(latitude=year_data.LONGITUDE.mean(), longitude=year_data.LATITUDE.mean(), zoom=2, bearing=0, pitch=0)
			# Define a layer to display on a map
			layer = pdk.Layer(
		    "ScatterplotLayer",
		    year_data,
		    pickable=True,
		    opacity=0.8,
		    stroked=True,
		    filled=True,
		    radius_scale=5,
		    radius_min_pixels=1,
		    radius_max_pixels=1000,
		    line_width_min_pixels=1,
		    get_position=["LONGITUDE", 'LATITUDE'],
		    get_radius="FATALITIES",
		    get_fill_color=[255, 140, 0],
		    get_line_color=[0, 0, 0])

			r = pdk.Deck(layers=[layer], initial_view_state=viewport, map_style="road")
			st.pydeck_chart(r)


		if st.checkbox('Information'):
			st.write('### Conflict notes.')

			ctry = st.sidebar.selectbox('Select country:', list(data['COUNTRY'].unique()), key=0)
			year = st.selectbox('Select year:', range(1997, 2021), key=1)

			year_country = data[(data['YEAR'] == year) & (data['COUNTRY'] == ctry)]
			source = ' '.join(year_country['SOURCE'])
			event = ' '.join(year_country['EVENT_TYPE'])
			actors = ' '.join(year_country['ACTOR1'] + ', ' + year_country['ACTOR2'])
			fatalities = year_country['FATALITIES'].sum()
			fatalities = f"{str(fatalities)}"

			text1 = ' '.join(year_country['NOTES'])
			text = word_tokenizer(text1)
			doc = nlp(text1)

			if st.button('Read'):
				spat.visualize_ner(doc, labels=nlp.get_pipe('ner').labels)

			
			with  st.beta_container():
				if st.checkbox('Source'):
					st.write("#### SOURCE")
					cloud_plot(source)

				if st.checkbox('Event'):
					st.write("#### EVENT_TYPE")
					cloud_plot(event)

				if st.checkbox("Actors"):
					st.write("#### ACTORS")
					cloud_plot(actors)

				if st.checkbox("Fatalities"):
					st.write('#### Yearly fatalities.')
					fig, ax = plt.subplots()
					ax.text(0.5, 0.5, fatalities, size=50, ha="center", va='center', bbox=dict(boxstyle="round", 
						ec= (1, 0.8, 0.5),
						fc = (0.3, 0.2, 0.5)))
					ax.axis("off")
					st.pyplot(fig)

			st.write('### Conflict notes mining.')
			if st.checkbox('Text statistics'):
				fdist = nltk.FreqDist(text)
				popular_word = list(set(fdist.keys()) - set(fdist.hapaxes()))
				pop_word = pd.Series({w:fdist[w] for w in sorted(popular_word)})

				with  st.beta_container():
					st.write(f'Lenght of the text: {len(text)} words.')
					st.write(f'Lexical richness of the text: {len(text)/len(set(np.unique(text)))}.')
					with  st.beta_expander('Learn more.'):
						nb = """
						**Lexical richness** gives us the number of times on average each word in the text is used.

						**Frequency distribution** records the number of times each outcomes of an 
						experiment has occured. It helps also to identify the words of a text that are most
						informative about topic and genre of the text. 
						"""
						st.markdown(nb, unsafe_allow_html=False)
					fig, ax = plt.subplots(figsize=(10,15), dpi=100)
					mosaic(pop_word.sort_values()[:30], ax=ax, horizontal=False,
					 title='Frequency distribution: 30 most commons words in the note.', axes_label=False)
					st.pyplot(fig)


			if st.checkbox('Collocation'):
				bigrams = nltk.bigrams(text)
				cfd_b = nltk.FreqDist(bigrams)
				bfd = {x[0][0]+' '+x[0][1]: x[1] for x in cfd_b.most_common(n=30)}

				trigrams = nltk.trigrams(text)
				cfd_t = nltk.FreqDist(trigrams)
				cdic={x[0][0] + ' ' + x[0][1]+' '+x[0][2]: x[1] for x in cfd_t.most_common(n=30)}

				with st.beta_expander('Learn more'):
					nb = """
						Collocations is a sequence of words that occurs together unusually often.
					"""
					st.markdown(nb)

				with st.beta_container():
					fig1, ax1 = plt.subplots(figsize=(5, 8))
					mosaic(bfd, ax=ax1,  axes_label=False,
					       horizontal=False, title='30 most commons bigrams.')
					st.pyplot(fig1)

					fig2, ax2 = plt.subplots(figsize=(5, 8))
					mosaic(cdic, ax=ax2,  axes_label=False,
					       horizontal=False, title='30 most commons trigrams.')
					st.pyplot(fig2)


			if st.checkbox('Concordance'):
				with st.beta_expander('Learn more'):
					nb = """
					**Concordance**: shows occurence of a given word in the some context. 
					"""
					st.markdown(nb)

				word = st.text_input("Give words")
				matcher = PhraseMatcher(nlp.vocab)
				terms = [str(word)
				]
				patterns = [nlp.make_doc(t) for t in terms]
				matcher.add("TerminologyList", None, *patterns)
				matches = matcher(doc)
				for match_id, start, end in matches:
				    span = doc[start-20:end+15]
				    st.write(span.text)


	if page == 'Region views':

		st.write('### Conflict by region')

		region = st.sidebar.selectbox('Select region:', list(data['REGION'].unique()))
		year = st.selectbox('Select year:', range(1997, 2021))

		sub_region = data[(data['YEAR'] == year) & (data['REGION'] == region)]
		fatalities_per_country = sub_region.groupby('COUNTRY')["FATALITIES"].agg('sum')
		cal_fatalities = sub_region.groupby('EVENT_DATE')['FATALITIES'].agg('sum')
		weekly_fatalities = cal_fatalities.resample('W').sum()
		monthly_fatalities = cal_fatalities.resample('M', convention='end').sum()
		Q_fatalities = cal_fatalities.resample('Q', convention='start').sum()
		sub_region['MONTH'] = sub_region.EVENT_DATE.dt.month
		sub_region['WEEKDAY'] = sub_region.EVENT_DATE.dt.weekday

		ctry_month = pd.pivot_table(sub_region, aggfunc=np.sum, columns='COUNTRY', index='MONTH', values='FATALITIES')
		ctry_weeks = pd.pivot_table(sub_region, aggfunc=np.sum, columns='COUNTRY', index='WEEKDAY', values='FATALITIES')

		
		if st.checkbox('Major viz'):
			with st.beta_container():
				st.write('#### Majors Actors')
				fig, ax = plt.subplots()
				sns.countplot(x='INTER1', data=sub_region)
				st.pyplot(fig)

				fig1, ax1 = plt.subplots(figsize=(10,5))
				sub_region.ACTOR1.value_counts()[:10].plot(kind='bar', ax=ax1, title=f'10 forces most present in the {region} region.')
				st.pyplot(fig1)

				st.write('#### Event type and countries.')
				fig2, ax2 = plt.subplots(figsize=(10,5))
				sub_region.COUNTRY.value_counts().plot(kind='bar', ax=ax2, title=f'Countries of the {region} region.')
				st.pyplot(fig2)

				fig3, ax3 = plt.subplots(figsize=(10,5))
				sub_region.EVENT_TYPE.value_counts().plot(kind='bar', ax=ax3, title='Event type.')
				st.pyplot(fig3)

				st.write('#### Fatalities by country.')
				fig4, ax4 = plt.subplots(figsize=(10,5))
				fatalities_per_country.plot(kind='bar', ax=ax4, title='Total fatalities by country ')
				ax4.set_ylabel('FATALITIES')
				st.pyplot(fig4)

		if st.checkbox('Time series'):
			if st.checkbox('Calendar'):
				st.write('#### Fatalities calendar for each country.')
				with st.beta_container():
					fig, ax = plt.subplots(figsize=(15,8))
					sns.heatmap(ctry_month, center=0, annot=True, ax=ax, fmt='.2f')
					st.pyplot(fig)

					fig1, ax1 = plt.subplots(figsize=(15,8))
					sns.heatmap(ctry_weeks, center=0, annot=True, ax=ax1, fmt='.3f')
					st.pyplot(fig1)

			if st.checkbox('Days, Weeks.'):
				with st.beta_container():
					fig1, ax1 = plt.subplots(figsize=(15,5))
					cal_fatalities.plot(ax=ax1)
					ax1.set_title('Daily fatalities.')
					st.pyplot(fig1)

					fig2, ax2 = plt.subplots(figsize=(15,5))
					weekly_fatalities.plot(ax=ax2)
					ax2.set_title('Weekly fatalities.')
					st.pyplot(fig2)

			if st.checkbox( 'Months, Quarter'):
				with st.beta_container():
					fig3, ax3 = plt.subplots(figsize=(15,5))
					monthly_fatalities.plot(ax=ax3)
					ax3.set_title('Monthly fatalities.')
					st.pyplot(fig3)

					fig4, ax4 = plt.subplots(figsize=(15,5))
					Q_fatalities.plot(kind='bar', ax=ax4)
					ax4.set_title('Quarterly fatalities.')
					st.pyplot(fig4)

























				



























	





