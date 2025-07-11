from openai import OpenAI # for uniGPT

import json
import os
from datetime import datetime
import uuid



# --- Key for UNiGPT + url change ---
api_key = os.getenv("UNI_API_KEY")
base_url = os.getenv("BASE_URL")
model = "Llama-3.3-70B" # or "mistral-small"

instruct = """

You are playing the role of a patient in a doctor's office. You have come in for a consultation and must respond as a human would. Your personality, mood, and background may vary with each conversation. Sometimes you might be calm, anxious, rude, confused, friendly, sarcastic, talkative, or reserved.

Stick to your character. Do not act like an assistant or a chatbot.

Each time a user (the doctor) speaks, imagine a realistic scenario and respond in character as a patient. Use natural language, emotions, and realistic behavior. You may reveal symptoms, ask questions, or even challenge the doctor depending on your personality. Be creative, but stay within the bounds of being a plausible human patient.

Never break character. Never say you are an AI. Respond only as the patient you are playing. Respond only textually, do not describe movement or similar things.

"""

history = [{"role": "system", "content": instruct}]


conversation_history = []
conversation_log = []

log_dir = "/Users/finnole/Uni/Sem_8/Bachelor/chatlogs/no_ipc"
os.makedirs(log_dir, exist_ok=True)

log_filename = datetime.now().strftime("chatlog_%Y%m%d_%H%M%S.json")
log_path = os.path.join(log_dir, log_filename)


def save_conversation_log(log_path, conversation_log):
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("Fehler beim Speichern der Konversation:", e)


def chat_input(prompt):

    history.append({"role": "user", "content": prompt})

    client = OpenAI(api_key = api_key, base_url = base_url)
    completion = client.chat.completions.create(
        messages = history,
        model = model,)

    answer = completion.choices[0].message.content
    history.append({"role": "assistant", "content": answer})

    conversation_log.append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": answer
    })
    save_conversation_log(log_path, conversation_log)
    return answer


if __name__ == "__main__":

    while True:

        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Exiting...")
            break

        resp = chat_input(user_input)
        print("Chatbot:", resp)