# 🎬 Emotion-Based Movie Recommendation System

This is an intelligent movie recommender system that detects the user's mood from their facial expression using a webcam and recommends personalized movies based on their emotional state. It leverages facial emotion recognition and BERT-based semantic understanding to suggest relevant movies, all wrapped in a Streamlit web application.

---

## 🚀 Features

- 🎭 **Real-time Mood Detection**: Uses OpenCV and Haar Cascade to detect faces and classify emotions.
- 🤖 **Context-Aware Recommendations**: Uses BERT embeddings from movie metadata (overview + genres + keywords) to compute semantic similarity.
- 🎯 **Personalized Results**: Matches the user's mood to semantically related movie plots using cosine similarity.
- 🌐 **Web App Interface**: Built with Streamlit for interactive user experience.

---

## 🛠️ Tools & Technologies

- `Python`
- `OpenCV` – for face detection and webcam streaming  
- `Haar Cascade` – for facial feature detection  
- `Pretrained Emotion Classifier` – to detect emotions like Happy, Sad, Angry, etc.  
- `Hugging Face Transformers (BERT)` – to embed movie descriptions semantically  
- `PyTorch` – for running BERT on GPU  
- `Pandas` / `NumPy` – for data handling  
- `Streamlit` – for frontend and deployment  
- `CUDA` – for GPU acceleration  

---

## 📁 Dataset Used

- [TMDB 5000 Movies Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)  
  Used metadata including:
  - `genres`
  - `keywords`
  - `overview`
  - `title`
  - `popularity`, `vote_average`, `vote_count`

---

## 🔍 How It Works

1. **Facial Emotion Detection**  
   The webcam captures an image → Haar Cascade locates the face → A trained emotion model classifies the user's emotion.

2. **Movie Embedding with BERT**  
   Movie metadata is preprocessed and embedded using `bert-base-uncased`. Pooler output vectors are extracted for each movie.

3. **Recommendation via Semantic Similarity**  
   The user’s mood is mapped to a predefined text (e.g., “I am feeling happy”) and embedded. Cosine similarity is used to find the most relevant movie vectors.

4. **Display via Streamlit**  
   The top recommendations are shown on a Streamlit app with titles and descriptions.

---

## ▶️ Running the App

```bash
# 1. Clone the repo
git clone https://github.com/Partha-png/movierecommendersystem.git
cd emotion-movie-recommender

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
