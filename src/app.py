# app.py

import streamlit as st
from main import recommend_movies

st.title("🎬 Mood-based Movie Recommender")

if st.button("Detect Mood & Recommend Movies"):
    with st.spinner('Detecting mood and finding best movies...'):
        mood, recommendations = recommend_movies()
    st.success(f"**Detected Mood:** `{mood}`")
    st.subheader("🎥 Top Recommendations:")
    for movie in recommendations:
        st.write(f"✅ {movie}")