import os
import nltk
import numpy as np
import tensorflow as tf
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder

# ✅ Set Correct NLTK Data Path
nltk.data.path.append("/Users/mahakal/nltk_data")

# ✅ Load 'punkt_tab' Instead of 'punkt'
try:
    nltk.data.find("tokenizers/punkt_tab")
    print("✅ 'punkt_tab' tokenizer found!")
except LookupError:
    nltk.download("punkt_tab", download_dir="/Users/mahakal/nltk_data")

try:
    nltk.data.find("corpora/stopwords")
    print("✅ 'stopwords' corpus found!")
except LookupError:
    nltk.download("stopwords", download_dir="/Users/mahakal/nltk_data")

stop_words = set(stopwords.words("english"))

# 📌 Sample chatbot training data
training_data = [
    {"input": "Hello", "output": "Hi! How can I help you?"},
    {"input": "How are you?", "output": "I'm just a chatbot, but I'm doing great!"},
    {"input": "Tell me a joke", "output": "Why don’t scientists trust atoms? Because they make up everything!"},
    {"input": "What is your name?", "output": "I am an AI chatbot!"},
    {"input": "What is AI?", "output": "AI stands for Artificial Intelligence."},
    {"input": "What can you do?", "output": "I can answer questions and assist you!"}
]

# 📌 Tokenize & Clean Input Text
inputs = [
    [word.lower() for word in word_tokenize(sample["input"]) if word.isalnum() and word.lower() not in stop_words]
    for sample in training_data
]
outputs = [sample["output"] for sample in training_data]

# 📌 Encode Text Labels into Numeric Values
encoder = LabelEncoder()
labels = encoder.fit_transform(outputs)

# 📌 Convert Input Words to Numerical Format
max_input_length = max(len(seq) for seq in inputs)
X_train = np.zeros((len(inputs), max_input_length), dtype=np.float32)

for i, seq in enumerate(inputs):
    for j, word in enumerate(seq):
        X_train[i, j] = hash(word) % 1000  # Convert words to numerical values

y_train = np.array(labels)

# 📌 Build Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, input_shape=(max_input_length,), activation="relu"),
    tf.keras.layers.Dense(16, activation="relu"),
    tf.keras.layers.Dense(len(set(outputs)), activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

# 📌 Train the Model
model.fit(X_train, y_train, epochs=10, verbose=1)

# 📌 Save the Trained Model
model.save("chatbot_model.h5")

print("🎉 Model training complete. Saved as chatbot_model.h5!")

