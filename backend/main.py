from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ✅ Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Ensure NLTK resources are available inside Docker
import os
nltk_data_path = "/usr/local/nltk_data"
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

try:
    nltk.data.find("tokenizers/punkt")
    print("✅ 'punkt' tokenizer found!")
except LookupError:
    nltk.download("punkt", download_dir=nltk_data_path)

# ✅ Load trained chatbot model
try:
    model = tf.keras.models.load_model("chatbot_model.h5")
    print("✅ Chatbot model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")

class ChatRequest(BaseModel):
    message: str

def preprocess_text(message):
    """Convert input text to numerical representation"""
    words = word_tokenize(message.lower())  # Tokenize message
    numerical_data = np.zeros((1, model.input_shape[1]), dtype=np.float32)  # Create zero array

    for j, word in enumerate(words):
        numerical_data[0, j] = hash(word) % 1000  # Convert words to numerical values

    return numerical_data

@app.post("/chat/")
def chat_response(request: ChatRequest):
    try:
        # ✅ Convert text to numerical format
        input_data = preprocess_text(request.message)
    
        # ✅ Predict response
        predictions = model.predict(input_data)
        response_index = np.argmax(predictions)

        responses = [
            "Hi! How can I help you?",
            "I'm just a chatbot!",
            "I am an AI chatbot!",
            "AI stands for Artificial Intelligence.",
            "I can answer questions and assist you!"
        ]
        return {"response": responses[response_index]}

    except Exception as e:
        return {"error": f"Something went wrong: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

