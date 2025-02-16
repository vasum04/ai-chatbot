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
    nltk.data.find("tokenizers/punkt_tab")
    print("✅ 'punkt_tab' tokenizer found!")
except LookupError:
    nltk.download("punkt_tab", download_dir=nltk_data_path)

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
    max_length = model.input_shape[1] if model else 10  # Ensure correct input shape
    numerical_data = np.zeros((1, max_length), dtype=np.float32)

    for j, word in enumerate(words[:max_length]):  # Limit input length
        numerical_data[0, j] = hash(word) % 1000  # Convert words to numerical values

    return numerical_data

@app.post("/chat/")
def chat_response(request: ChatRequest):
    try:
        if model is None:
            return {"error": "Model not loaded properly"}

        input_data = preprocess_text(request.message)
        predictions = model.predict(input_data)
        response_index = np.argmax(predictions)  # Get the index of the highest$
        
        print(f"Model Predictions: {predictions}, Selected Index: {response_index}")  # Debugging output


        response_map = {
            "hello": "Hi! How are you doing today?",
            "hi": "Hello! How can I assist you?",
            "how are you": "I'm just a chatbot, but I'm here to help!",
            "who are you": "I'm an AI chatbot designed to assist you.",
            "what is ai": "AI stands for Artificial Intelligence.",
            "bye": "Goodbye! Have a great day!"
        }

        message_cleaned = request.message.lower().strip()
        response = response_map.get(message_cleaned, "I'm sorry, I didn't understand that.")

        return {"response": response}
    
    except Exception as e:
        return {"error": f"Something went wrong: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
