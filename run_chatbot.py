import json
import numpy as np
import random
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

# Load Dataset (for responses and tags)
with open("intents.json", "r") as file:
    data = json.load(file)

# Preprocessing setup (must match training)
lemmatizer = WordNetLemmatizer()
patterns = []
tags = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words = nltk.word_tokenize(pattern.lower())
        words = [lemmatizer.lemmatize(word) for word in words]
        patterns.append(" ".join(words))
        tags.append(intent["tag"])

# Recreate vectorizer and label encoder (must match training)
vectorizer = TfidfVectorizer()
vectorizer.fit(patterns)  # Fit on the same patterns used during training
label_encoder = LabelEncoder()
label_encoder.fit(tags)   # Fit on the same tags used during training

# Load the trained model
model = load_model('chatbot_model.h5')
print("Model loaded successfully")

# Chatbot Function
def chatbot_response(user_input):
    processed_input = " ".join([lemmatizer.lemmatize(word) for word in nltk.word_tokenize(user_input.lower())])
    X_input = vectorizer.transform([processed_input]).toarray()
    tag_index = np.argmax(model.predict(X_input, verbose=0))
    tag = label_encoder.inverse_transform([tag_index])[0]
    response = random.choice([resp for intent in data["intents"] if intent["tag"] == tag for resp in intent["responses"]])
    return response

# Chat Loop
print("Chatbot is ready! Type 'quit' to exit.")
while True:
    user_input = input("You: ")
    if user_input.lower() == 'quit':
        break
    print("Bot:", chatbot_response(user_input))