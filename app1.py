#!/usr/bin/python
# -*- coding: utf-8 -*-
import streamlit as st
import joblib
import os
import spacy
import pandas as pd
from newspaper import Article
nlp = spacy.load('en_core_web_sm')
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import nltk 
nltk.download('punkt')


news_vectorizer = open('models/final_news_cv_vectorizer.pkl', 'rb')
news_cv = joblib.load(news_vectorizer)


def load_prediction_models(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model
    
def get_article_details(url):
    article = Article(url)
    article.download()
    article.parse()
    article.nlp()
    text = article.text
    summary = article.summary
    return text

def get_key(val, my_dict):
    for (key, value) in my_dict.items():
        if val == value:
            return key

def main():
    """News Classifier"""

    st.title("""News Classifier""")

    # st.subheader("ML App with Streamlit")

    html_temp = \
        """
	<div style="background-color:blue;padding:10px">
	<h1 style="color:white;text-align:center;">News Classification WebApp </h1>
	</div>
	"""
    st.markdown(html_temp, unsafe_allow_html=True)

    activity = ['Prediction', 'NLP']
    choice = st.sidebar.selectbox('Select Activity', activity)

    if choice == 'Prediction':
        st.info('Prediction with ML')

        news_text_ = st.text_area('Enter News URL Here', 'Paste URL Here.')
        all_ml_models = ['LR', 'RFOREST', 'NB', 'DECISION_TREE']
        model_choice = st.selectbox('Select Model', all_ml_models)

        prediction_labels = {
            'business': 0,
            'tech': 1,
            'sport': 2,
            'health': 3,
            'politics': 4,
            'entertainment': 5,
            }
        if st.button('Classify'):
            news_text = get_article_details(news_text_)
            st.text('Original Text::\n{}'.format(news_text))
            vect_text = news_cv.transform([news_text]).toarray()
            if model_choice == 'LR':
                predictor = \
                    load_prediction_models('models/newsclassifier_Logit_model.pkl'
                        )
                prediction = predictor.predict(vect_text)
            elif model_choice == 'RFOREST':

                # st.write(prediction)

                predictor = \
                    load_prediction_models('models/newsclassifier_RFOREST_model.pkl'
                        )
                prediction = predictor.predict(vect_text)
            elif model_choice == 'NB':

                # st.write(prediction)

                predictor = \
                    load_prediction_models('models/newsclassifier_NB_model.pkl'
                        )
                prediction = predictor.predict(vect_text)
            elif model_choice == 'DECISION_TREE':

                # st.write(prediction)

                predictor = \
                    load_prediction_models('models/newsclassifier_CART_model.pkl'
                        )
                prediction = predictor.predict(vect_text)

                # st.write(prediction)

            final_result = get_key(prediction, prediction_labels)
            st.success('News Categorized as:: {}'.format(final_result))

    if choice == 'NLP':
        st.info('Natural Language Processing of Text')
        raw_text_ = st.text_area('Enter News URL Here', 'Paste URL Here')
        nlp_task = ['Tokenization', 'Lemmatization', 'NER', 'POS Tags']
        task_choice = st.selectbox('Choose NLP Task', nlp_task)
        if st.button('Analyze'):
            raw_text = get_article_details(raw_text_)
            st.info('Original Text::\n{}'.format(raw_text))

            docx = nlp(raw_text)
            if task_choice == 'Tokenization':
                result = [token.text for token in docx]
            elif task_choice == 'Lemmatization':
                result = ["'Token':{},'Lemma':{}".format(token.text,
                          token.lemma_) for token in docx]
            elif task_choice == 'NER':
                result = [(entity.text, entity.label_) for entity in
                          docx.ents]
            elif task_choice == 'POS Tags':
                result = \
                    ["'Token':{},'POS':{},'Dependency':{}".format(word.text,
                     word.tag_, word.dep_) for word in docx]

            st.json(result)

        if st.button('Tabulize'):
            raw_text = get_article_details(raw_text_)
            docx = nlp(raw_text)
            c_tokens = [token.text for token in docx]
            c_lemma = [token.lemma_ for token in docx]
            c_pos = [token.pos_ for token in docx]

            new_df = pd.DataFrame(zip(c_tokens, c_lemma, c_pos),
                                  columns=['Tokens', 'Lemma', 'POS'])
            st.dataframe(new_df)

        if st.checkbox('WordCloud'):
            raw_text = get_article_details(raw_text_)
            c_text = raw_text
            wordcloud = WordCloud().generate(c_text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot()

    st.sidebar.subheader('About')


if __name__ == '__main__':
    main()
