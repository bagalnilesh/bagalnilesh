import pandas as pd
import random 
from faker import Faker

fake = Faker()

data = []
for _ in range(100):
    source = random.choice(['Email','Twitter','Google Review','Survey'])
    sentiment = random.choice(['Positive','Negative','Neutral'])

    if sentiment == 'positive':
        text = fake.sentence(nb_words = 15) + "great amazing love you"
    elif sentiment == 'negative':
        text = fake.sentence(nb_words = 15) + "Bad Terrible Plane crash in Ahmedabad"
    else:
        text = fake.sentence(nb_words =20)
    data.append({
        'date': fake.date_between(start_date='-3m'),
        'source': source,
        'text': text,
        'product': random.choice(['mobile app','website','customer service']),
        'customer_id': fake.uuid4()[:8]
    })
df = pd.DataFrame(data)
df.to_csv('dummy_customer_feedback.csv')

# Sentiment Analysis Pipeline
from transformers import pipeline
sentiment_analyzer = pipeline("text-classification",model = "distilbert-base-uncased-finetuned-sst-2-english")
sample_text = "The App crashes frequently but I love customer service"
result =  sentiment_analyzer(sample_text)
print(f"Sentiment: {result[0]['label']},Confidence: {result[0]['score']:.2f}")

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


vectorizer = CountVectorizer(max_df=0.95, min_df=2, stop_words='english')
dtm = vectorizer.fit_transform(df['text'])


lda = LatentDirichletAllocation(n_components=5)
lda.fit(dtm)

# Show top words per topic
for index, topic in enumerate(lda.components_):
    print(f"Topic #{index}:")
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])


import re

def detect_urgency(text):
    urgent_keywords = ['emergency', 'refund now', 'legal action', 'terrible', 'sue']
    if re.search('|'.join(urgent_keywords), text.lower()):
        return "High Priority"
    elif 'disappoint' in text.lower() or 'angry' in text.lower():
        return "Medium Priority"
    return "Normal"
   
df['priority'] = df['text'].apply(detect_urgency)
print("***")

import streamlit as st
import pandas as pd
import plotly.express as px


# df = pd.read_csv('dummy_customer_feedback.csv')

@st.cache_data
def load_data():
    return pd.read_csv('dummy_customer_feedback.csv')
df = load_data()
if 'sentiment' not in df.columns:
    from textblob import TextBlob
    df['sentiment'] = df['text'].apply(lambda x:'Positive' if TextBlob(x).sentiment.polarity>0 else 'Negative')

st.title("Customer Sentiment Dashboard")
fig1 = px.pie(df, names='sentiment', title='Overall Sentiment')
st.plotly_chart(fig1)

st.subheader("Feedback Sources")
fig2 = px.bar(df['source'].value_counts(),title = "Feedback by channel")
st.plotly_chart(fig2)
# topics = ["Login Issues", "Payment Problems", "Feature Requests", "Shipping", "Pricing"]
# st.write(", ".join(topics))

# st.subheader("Urgent Complaints")
# st.table(df[df['priority'] == 'High Priority'].head(3))

