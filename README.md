# Emotion-Based Movie Recommendation System

This project is an intelligent movie recommender system that detects the user's mood from their facial expression using a webcam and recommends personalized movies based on their emotional state.  
It leverages **facial emotion recognition** and **BERT-based semantic understanding** to suggest relevant movies, all wrapped in a **Streamlit web application**.

---

## Features

- **Real-time Mood Detection:** Uses OpenCV and Haar Cascade to detect faces and classify emotions.  
- **Context-Aware Recommendations:** Uses BERT embeddings from movie metadata (overview + genres + keywords) to compute semantic similarity.  
- **Personalized Results:** Matches the user's mood to semantically related movie plots using cosine similarity.  
- **Web App Interface:** Built with Streamlit for interactive user experience.  

---

## Tools & Technologies

- **Language:** Python  
- **Computer Vision:** OpenCV, Haar Cascade  
- **Emotion Recognition:** Pretrained emotion classifier (Happy, Sad, Angry, etc.)  
- **NLP:** Hugging Face Transformers (BERT)  
- **Deep Learning:** PyTorch (GPU-enabled)  
- **Data Handling:** Pandas, NumPy  
- **Frontend & Deployment:** Streamlit  
- **Acceleration:** CUDA for GPU support  

---

## Dataset Used

**TMDB 5000 Movies Dataset**  
The following metadata fields are used:  
- genres  
- keywords  
- overview  
- title  
- popularity, vote_average, vote_count  

---

## How It Works

1. **Facial Emotion Detection**  
   - The webcam captures an image.  
   - Haar Cascade locates the face.  
   - A trained emotion model classifies the user's emotion.  

2. **Movie Embedding with BERT**  
   - Movie metadata is preprocessed and embedded using `bert-base-uncased`.  
   - Pooler output vectors are extracted for each movie.  

3. **Recommendation via Semantic Similarity**  
   - The user’s mood is mapped to a predefined text (e.g., *"I am feeling happy"*).  
   - This mood text is embedded using BERT.  
   - Cosine similarity is computed against movie vectors to retrieve the most relevant matches.  

4. **Display via Streamlit**  
   - The top recommendations are shown with movie titles and descriptions in an interactive web app.  

---

## Running the App

```bash
# 1. Clone the repository
git clone https://github.com/Partha-png/movierecommendersystem.git
cd emotion-movie-recommender

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate       # On Windows
source venv/bin/activate    # On Linux/MacOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py
```
#Project Structure
movierecommendersystem/
│
├── data/                         # Dataset folder
│   ├── test/                     # Test split of TMDB dataset
│   ├── train/                    # Training split of TMDB dataset
│   └── tmdb_5000_movies.csv      # Raw movie metadata
│
├── scripts/
│   └── main.ipynb                # Jupyter notebook for experiments
│
├── src/                          # Core source code
│   ├── app.py                    # Streamlit app entry point
│   ├── camera_detect.py          # Emotion detection from webcam
│   ├── camera.py                 # Camera utilities
│   └── main.py                   # Main pipeline script
│
├── templates/                    # HTML templates (if used by Streamlit/Flask)
│
├── venv/                         # Virtual environment
│
├── model_file.h5                 # Saved emotion detection model
├── data.yaml                     # Data configuration
├── requirements.txt              # Dependencies
├── README.md                     # Project documentation
└── .gitignore                    # Git ignore file
#Future Work

Improve emotion classification with deep CNN/ViT models.

Extend to multi-modal input (text + audio + video).

Deploy the Streamlit app on Hugging Face Spaces or AWS.

Expand the movie database with live TMDB API integration.

#Citation & Inspiration

TMDB 5000 Movies Dataset

Hugging Face Transformers

OpenCV + Haar Cascade Face Detection

Streamlit Documentation
