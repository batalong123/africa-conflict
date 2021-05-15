import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st
import joblib
from module.tokenization import tokenizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def predict():

	classifier = joblib.load('models/Event_type_TextClassification.sav')# classifier
	classifier1 = joblib.load('models/Region_TextClassification.sav')

	st.header('What event is it and where is occurred?')
	
	with st.beta_expander('Learn more'):
		st.markdown(
			"""
		Event are encoded like this:

		1. 0 --> Battles.
		2. 1 --> Explosions/Remote violence.
		3. 2 --> Protests.
		4. 3 --> Riots.
		5. 4 --> Strategic developments.
		6. 5 --> Violence against civilians.

		Region are encoded like this:

		1. 0 --> Eastern Africa
		2. 1 --> Middle Africa
		3. 2 --> Northern Africa
		4. 3 --> Southern Africa
		5. 4 --> Western Africa
			""")



	note = []
	nb_text = st.number_input('Give number of text.',1)

	for i in range(int(nb_text)):
		
		note.append(st.text_area('Give source text only one:', key=i)) # note write by source media.
	
	note = np.array(note)

	
	#raw_file = st.text_input('Give csv file text path:')#load csv file
	st.write(note)

	label = ['Battles', 'Explosions/Remote violence', 'Protests', 'Riots', 'Strategic developments', 'Violence against civilians']
	label1 = ['Eastern Africa', 'Middle Africa', 'Northern Africa', 'Southern Africa', 'Western Africa']


	st.subheader('What event type is it?')
	if st.button('predict', key=0):

		#ext = os.path.splitext(raw_file)[1]
		if nb_text > 1:

			proba = classifier.predict_proba(note)
			textp = ['probability of text'+str(i) for i in range(int(nb_text))]
			prob = pd.DataFrame(proba, columns=label, index=textp)
			st.success('Prediction is ok, see probability.')
			st.dataframe(prob)

		else:

			pred = classifier.predict(note)
			proba = classifier.predict_proba(note)
			res = f'Event is {label[pred[0]]} with the probability of {100*proba[0][pred][0]}%'

			prob = pd.DataFrame(100*proba[0], columns=['probability(%)'], index=label)
			st.success(res)
			st.dataframe(prob)


	st.subheader('Where the event takes places?:')
	if st.button('predict', key=1):

		#ext = os.path.splitext(raw_file)[1]
		if nb_text > 1:

			proba = classifier1.predict_proba(note)
			textp = ['probability of text'+str(i) for i in range(int(nb_text))]
			prob = pd.DataFrame(proba, columns=label1, index=textp)
			st.success('Prediction is ok, see probability.')
			st.dataframe(prob)

		else:

			pred = classifier1.predict(note)
			proba = classifier1.predict_proba(note)
			res = f'Event is {label1[pred[0]]} with the probability of {100*proba[0][pred][0]}%'

			prob = pd.DataFrame(100*proba[0], columns=['probability(%)'], index=label1)
			st.success(res)
			st.dataframe(prob)
