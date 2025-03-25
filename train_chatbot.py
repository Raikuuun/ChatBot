import json
import numpy as np
import random
import nltk
nltk.download('punkt')      # For tokenization
nltk.download('wordnet')    # For lemmatization
import tensorflow as tf
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import os
import platform

# Disable oneDNN to avoid OverflowError
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
print("TF_ENABLE_ONEDNN_OPTS:", os.environ.get('TF_ENABLE_ONEDNN_OPTS'))

# Set float32 and determinism with a seed
tf.keras.backend.set_floatx('float32')
tf.config.experimental.enable_op_determinism()
tf.random.set_seed(1)


# Load Dataset
with open("intents.json", "r") as file:
    data = json.load(file)

# Preprocessing
lemmatizer = WordNetLemmatizer()
patterns = []
tags = []
documents = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        words = nltk.word_tokenize(pattern.lower())
        words = [lemmatizer.lemmatize(word) for word in words]
        documents.append((words, intent["tag"]))
        patterns.append(" ".join(words))
        tags.append(intent["tag"])

# Convert text data to numerical format
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(patterns).toarray()
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(tags)

# Debug data
print("X shape:", X.shape)
print("X dtype:", X.dtype)
print("X sample:", X[0][:10])  # First 10 values of first sample
print("y shape:", y.shape)
print("y dtype:", y.dtype)
print("y sample:", y[0])

# Build Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(len(set(tags)), activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=200, batch_size=8, verbose=1)

# Save the model
model.save('chatbot_model.h5')

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