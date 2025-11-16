import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import os

# Download NLTK resources (runs only the first time)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
st.write("hello")
ps = PorterStemmer()   


st.title("📢 Email / SMS Spam Classifier")

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()
    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


# Load vectorizer + model
try:
    tfidf = pickle.load(open('vectorizer_new.pkl', 'rb'))
    model = pickle.load(open('model.pkl', 'rb'))
except Exception as e:
    st.error(f"🔥 Model loading failed: {e}")


input_sms = st.text_input("Enter the message")

if st.button("Predict"):   # 👈 Only predict when button clicked
    if input_sms.strip() != "":
        transform_sms = transform_text(input_sms)
        vector_input = tfidf.transform([transform_sms])
        result = model.predict(vector_input)[0]

        if result == 1:
            st.error("🚫 Spam Message")
        else:
            st.success("✔ Not Spam (Ham)")
    else:
        st.warning("Please enter a message first.")
