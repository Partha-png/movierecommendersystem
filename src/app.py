# app.py

import streamlit as st
from main import recommend_movies

try:
    st.title("🎬 Mood-based Movie Recommender")

    if st.button("Detect Mood & Recommend Movies"):
        with st.spinner('Detecting mood and finding best movies...'):
            mood, recommendations = recommend_movies()
        st.success(f"**Detected Mood:** `{mood}`")
        st.subheader("🎥 Top Recommendations:")
        for movie in recommendations:
            st.write(f"✅ {movie}")

except Exception:
    st.error("App crashed! See terminal for details.")
    traceback.print_exc(file=sys.stderr)