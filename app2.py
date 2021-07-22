# Core Pkgs
import streamlit as st 
import altair as alt
import plotly.express as px 

# EDA Pkgs
import pandas as pd 
import numpy as np 
from datetime import datetime

# Utils
import joblib 
pipe_lr = joblib.load(open("modelnlp.pkl","rb"))

# Track Utils
# from track_utils import create_page_visited_table,add_page_visited_details,view_all_page_visited_details,add_prediction_details,view_all_prediction_details,create_emotionclf_table

# # Googletrans unofficial
# from googletrans import Translator, constants
# from pprint import pprint
# translator = Translator(service_urls=["translate.google.co.id","translate.google.com"])

import tweepy
# api key
api_key = "l5FGlSMhD3FOB1phnwB7I9sX5"
# api secret key
api_secret_key = "R55gay8XG4uz1VGns8BT87zzXBGftNxPMaS9nvUVOzRI8YNsP1"
# access token
access_token = "237213820-RbW5PBW76TqcbT1tiAGjdkiMV7LlPnRIb9oDHixg"
# access token secret
access_token_secret = "EkHdik9UpmPB8CP8g3kSip0RC30LqgSRdkuGrovUnNEyN"
auth = tweepy.OAuthHandler(api_key,api_secret_key)
auth.set_access_token(access_token,access_token_secret)
api = tweepy.API(auth)


# Fxn
def predict_emotions(docx):
	results = pipe_lr.predict([docx])
	return results[0]

def get_prediction_proba(docx):
	results = pipe_lr.predict_proba([docx])
	return results

emotions_emoji_dict = {"anger":"😠","disgust":"🤮", "fear":"😨😱", "happy":"🤗", "joy":"😂", "neutral":"😐", "sad":"😔", "sadness":"😔", "shame":"😳", "surprise":"😮"}


# Main Application
# def main():
st.title("Emotion Classifier App")
menu = ["Home","NER"]
choice = st.sidebar.selectbox("Menu",menu)
# create_page_visited_table()
# create_emotionclf_table()
if choice == "Home":
	# add_page_visited_details("Home",datetime.now())
	st.subheader("Home-Emotion In Text")

	with st.form(key='emotion_clf_form'):
		search_text = st.text_area("Type Here")
		submit_text = st.form_submit_button(label='Submit')

	if submit_text:	
		hasilSearch = api.search(q=str(search_text),count=1)
		raw_text = []
		for tweet in hasilSearch:
			raw_text.append(tweet.text)
		raw_text = raw_text[0]
		col1,col2  = st.beta_columns(2)
		# translated = translator.translate(raw_text)
		# translated = raw_text
		prediction = predict_emotions(raw_text)
		probability = get_prediction_proba(raw_text)
		
		# add_prediction_details(raw_text,prediction,np.max(probability),datetime.now())

		with col1:
			st.success("Search Result")
			st.write(raw_text)
			# st.write(translated.text)
			st.success("Prediction")
			emoji_icon = emotions_emoji_dict[prediction]
			st.write("{}:{}".format(prediction,emoji_icon))
			st.write("Confidence:{}".format(np.max(probability)))

		with col2:
			st.success("Prediction Probability")
			# st.write(probability)
			proba_df = pd.DataFrame(probability,columns=pipe_lr.classes_)
			# st.write(proba_df.T)
			proba_df_clean = proba_df.T.reset_index()
			proba_df_clean.columns = ["emotions","probability"]

			fig = alt.Chart(proba_df_clean).mark_bar().encode(x='emotions',y='probability',color='emotions')
			st.altair_chart(fig,use_container_width=True)
