
import ollama
import pandas as pd
from datetime import datetime
import os

MODEL = "gemma3:1b"

history = []

# Files are saved inside the dataset folder
history_file = "dataset/chat_history.csv"
responses_file = "dataset/responses.csv"
logs_file = "dataset/chatbot_logs.txt"

print("=" * 45)
print("       WEEK 9 - AI CHATBOT")
print("       Powered by Ollama")
print("=" * 45)
print("Type 'exit' to stop.\n")

while True:

    user_input = input("You: ")

    if user_input.lower().strip() == "exit":
        print("Chatbot: Goodbye!")
        break

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    history.append({
        "role": "user",
        "content": user_input
    })

    try:
        response = ollama.chat(
            model=MODEL,
            messages=history
        )

        bot_reply = response["message"]["content"]

        print("Chatbot:", bot_reply)

        history.append({
            "role": "assistant",
            "content": bot_reply
        })

        record = {
            "timestamp": timestamp,
            "user_input": user_input,
            "chatbot_response": bot_reply
        }

        # Save conversation history
        pd.DataFrame([record]).to_csv(
            history_file,
            mode="a",
            header=not os.path.exists(history_file),
            index=False
        )

        # Save chatbot responses
        pd.DataFrame([record]).to_csv(
            responses_file,
            mode="a",
            header=not os.path.exists(responses_file),
            index=False
        )

        # Save text logs
        with open(logs_file, "a", encoding="utf-8") as file:
            file.write(f"[{timestamp}]\n")
            file.write(f"User: {user_input}\n")
            file.write(f"Chatbot: {bot_reply}\n\n")

    except Exception as e:
        print("Error:", e)

        # Remove unanswered user message from context
        history.pop()


