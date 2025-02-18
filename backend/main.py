from fastapi import FastAPI, Depends
import tensorflow as tf
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import os

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

# ✅ PostgreSQL Database Setup
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:password@db:5432/chatbot_db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ✅ Chat History Table
class ChatHistory(Base):
    __tablename__ = "chat_history"
    id = Column(Integer, primary_key=True, index=True)
    user_message = Column(String, nullable=False)
    bot_response = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

# ✅ Create the table in PostgreSQL
Base.metadata.create_all(bind=engine)

# ✅ Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ✅ Chat Request Model
class ChatRequest(BaseModel):
    message: str

# ✅ Text Preprocessing
def preprocess_text(message):
    words = word_tokenize(message.lower())
    max_length = model.input_shape[1] if model else 10
    numerical_data = np.zeros((1, max_length), dtype=np.float32)

    for j, word in enumerate(words[:max_length]):
        numerical_data[0, j] = hash(word) % 1000

    return numerical_data

# ✅ Chatbot API (Store in PostgreSQL)
@app.post("/chat/")
def chat_response(request: ChatRequest, db: Session = Depends(get_db)):
    try:
        if model is None:
            return {"error": "Model not loaded properly"}

        input_data = preprocess_text(request.message)
        predictions = model.predict(input_data)
        response_index = np.argmax(predictions)

        print(f"Model Predictions: {predictions}, Selected Index: {response_index}")

        # Predefined responses
        response_map = {
            "hello": "Hi! How are you doing today?",
            "hi": "Hello! How can I assist you?",
            "how are you": "I'm just a chatbot, but I'm here to help!",
            "who are you": "I'm an AI chatbot designed to assist you.",
            "what is ai": "AI stands for Artificial Intelligence.",
            "bye": "Goodbye! Have a great day!"
        }

        message_cleaned = request.message.lower().strip()
        bot_response = response_map.get(message_cleaned, "I'm sorry, I didn't understand that.")

        # ✅ Store chat in PostgreSQL
        chat_entry = ChatHistory(user_message=request.message, bot_response=bot_response)
        db.add(chat_entry)
        db.commit()

        return {"response": bot_response}

    except Exception as e:
        return {"error": f"Something went wrong: {str(e)}"}

# ✅ Fetch Chat History API
@app.get("/chat/history/")
def get_chat_history(db: Session = Depends(get_db)):
    chats = db.query(ChatHistory).order_by(ChatHistory.timestamp.desc()).all()
    return [{"id": chat.id, "message": chat.user_message, "response": chat.bot_response, "timestamp": chat.timestamp} for chat in chats]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
