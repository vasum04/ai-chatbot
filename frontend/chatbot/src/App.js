import React, { useState } from "react";
import axios from "axios";

function App() {
    const [message, setMessage] = useState("");
    const [response, setResponse] = useState("");

    const handleSend = async () => {
        try {
            const res = await axios.post("http://localhost:8000/chat/", 
                { message: message },  // Ensure JSON body is correct
                { headers: { "Content-Type": "application/json" } }  // Explicitly set headers
            );

            setResponse(res.data.response);
        } catch (error) {
            console.error("Error:", error);
            setResponse("Error communicating with backend");
        }
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

