import React, { useState } from "react";
import axios from "axios";

function App() {
    const [message, setMessage] = useState("");
    const [response, setResponse] = useState("");

    const handleSend = async () => {
        const res = await axios.post("http://127.0.0.1:8000/chat/", { message });
        setResponse(res.data.response);
    };

    return (
        <div>
            <h1>AI Chatbot</h1>
            <input
                type="text"
                value={message}
                onChange={(e) => setMessage(e.target.value)}
            />
            <button onClick={handleSend}>Send</button>
            <p>Response: {response}</p>
        </div>
    );
}

export default App;

