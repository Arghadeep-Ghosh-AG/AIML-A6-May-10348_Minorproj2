import streamlit as st
import joblib

model = joblib.load("fake_news_model.pkl")

st.title("Fake News Detector")

news_text = st.text_area(
    "Paste News Article",
    height=300
)

if st.button("Check News"):

    if news_text:

        prediction = model.predict(
            [news_text]
        )[0]

        st.success(
            f"Prediction: {prediction}"
        )
