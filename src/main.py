# main.py

from camera_detect import detect_emotion

import torch
from transformers import BertModel, BertTokenizer
import pandas as pd
import torch.nn.functional as F
import re

def recommend_movies():
    mood = detect_emotion()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to('cuda')
    model.eval()

    df = pd.read_csv(r'C:\Users\PARTHA SARATHI\Python\movierecommendersystem\data\tmdb_5000_movies.csv')
    df = df[['genres', 'keywords', 'overview', 'title', 'popularity', 'vote_average', 'vote_count']]
    df = df.dropna(axis=0)

    def preprocessing(row):
        genres = re.findall(r'"name":\s*"([^"]+)"', row['genres'])
        keywords = re.findall(r'"name":\s*"([^"]+)"', row['keywords'])
        overview = row['overview'].lower()
        title = row['title'].lower().replace(" ", "")
        tags = " ".join([*genres, *keywords]) + " " + overview + " " + title
        return tags.lower()

    df['tags'] = df.apply(preprocessing, axis=1)
    texts = df['overview'] + ' ' + df['tags']

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer.batch_encode_plus(
        texts.tolist(),
        add_special_tokens=True,
        max_length=128,
        truncation=True,
        padding='max_length',
        return_tensors='pt'
    )

    input_ids = encodings['input_ids'].to('cuda')
    attention_mask = encodings['attention_mask'].to('cuda')

    all_embeddings = []
    chunk_size = 128

    with torch.no_grad():
        for start in range(0, input_ids.size(0), chunk_size):
            end = start + chunk_size
            outputs = model(input_ids=input_ids[start:end], attention_mask=attention_mask[start:end])
            pooled_output = outputs.pooler_output
            all_embeddings.append(pooled_output.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0)
    normalized_embeddings = F.normalize(all_embeddings, p=2, dim=1)

    mood_encoding = tokenizer.encode(
        mood,
        add_special_tokens=True,
        max_length=16,
        truncation=True,
        return_tensors='pt'
    ).to('cuda')

    with torch.no_grad():
        mood_outputs = model(input_ids=mood_encoding)
        mood_embedding = mood_outputs.pooler_output.cpu()

    normalized_mood_embedding = F.normalize(mood_embedding, p=2, dim=1)
    similarities = torch.matmul(normalized_mood_embedding, normalized_embeddings.T).squeeze()
    top_k = torch.topk(similarities, k=5)
    recommended_indices = top_k.indices.tolist()

    recommendations = [df.iloc[i]['title'] for i in recommended_indices]

    return mood, recommendations
