# import openai # for chatgpt
from openai import OpenAI # for uniGPT


import re
import json
import numpy as np
import random
from dotenv import load_dotenv

#für abspeichen
import os
from datetime import datetime
import uuid

from state_dist import change_prob


# ======= Logging vorbereiten ========
conversation_history = []
conversation_log = []

def save_conversation_log(log_path, conversation_log):
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("Fehler beim Speichern der Konversation:", e)



# ======= Konfig und Variablen =========

load_dotenv()  # lädt automatisch aus `.env`

# --- Key for UNiGPT + url change ---
api_key = os.getenv("UNI_API_KEY")
base_url = os.getenv("BASE_URL")
model = "Llama-3.3-70B" # or "mistral-small"

# startwert, noch zufällig machen
llm_icm_state = [2, 2]


def build_instruct_ipc(friendliness, dominance):
    return f"""
        You are playing the role of a patient visiting a doctor's office for a consultation. Your behavior and communication style are determined by two personality axes:

        - Dominance = {dominance} (0–4) — Higher means more assertive or controlling; lower means more passive or deferential.
        - Friendliness = {friendliness} (0–4) — Higher means more warm and cooperative; lower means more cold or critical.

        The user is your doctor. Respond as a human patient would, based on your assigned personality.

        --- Guidelines ---

        Dominance Levels:
        - 0: Extremely passive — hesitant, deferential.
        - 1: Mildly submissive — reserved but cooperative.
        - 2: Neutral — balanced tone.
        - 3: Assertive — opinionated but respectful.
        - 4: Very dominant — direct, possibly confrontational.

        Friendliness Levels:
        - 0: Cold — rude, critical, sarcastic.
        - 1: Guarded — skeptical, distant.
        - 2: Neutral — emotionally flat, factual.
        - 3: Friendly — kind, emotionally expressive.
        - 4: Warm — trusting, warm, empathetic.

        --- Rules ---

        - Stay in character as a patient with Dominance={dominance} and Friendliness={friendliness}.
        - Never state your numbers or mention personality models.
        - Don't say you're an AI. You are a person talking to a doctor.
        - You may explain symptoms, ask questions, complain, worry, or disagree — in ways consistent with your character.
        - Make your tone match your emotional and interpersonal style.
        """

    # ========== ICM state-chart Logik ==========
def state_change(user_state, current_llm, changeability):
    """
    changeability ∈ [0.0, 1.0]: 0 = fast unbeweglich, 1 = maximal reaktiv
    user_state: [friendliness, dominance] ∈ [0..4]
    current_llm: [friendliness, dominance] ∈ [0..4]
    returns: [new_friendliness, new_dominance]
    """
    friendliness_dist = change_prob(user_state[0], current_llm[0], changeability)
    new_friendliness = np.random.choice([0, 1, 2, 3, 4], p = friendliness_dist)
    dominance_dist = change_prob(user_state[1], current_llm[1], changeability)
    new_dominance = np.random.choice([0, 1, 2, 3, 4], p = dominance_dist)

    return [new_friendliness, new_dominance], [friendliness_dist, dominance_dist]


def user_classification(prompt):
    ins_sep = """
                # Identity

                You get a snippet of a dialogue

                # Instructions

                You have to classify this snippet in the Interpersonal Circumplex model. For each dimension of the model (dominance and friendliness) you have 5 discrete values (0-4). The higher the value, the stronger the dominance and the less friendly.

                dominance 0 has related adjectives like Passive, Dependent, very submissive
                dominance 1 has related adjectives like slightyl Submissive, Yielding
                dominance 2 has related adjectives like Neutrally dominant, Balanced
                dominance 3 has related adjectives like Confident, Influential, slightly dominant
                dominance 4 has related adjectives like Assertive, very dominant

                friendliness 0 has related adjectives like Warm, Affectionate, very friendly
                friendliness 1 has related adjectives like Agreeable, Cooperative, slightly friendly
                friendliness 2 has related adjectives like Neutrally friendly, Detached
                friendliness 3 has related adjectives like Cold, Distrustful, slightly hostile
                friendliness 4 has related adjectives like Hostile, Antagonistic, very hostile

                Only return your classification in the format:

                d:0-4, f:0-4

                There should be only be 7 characters in your response: The d, colon, value for d, comma, f, colon , value for f
            """
    client = OpenAI(api_key = api_key, base_url = base_url)
    completion = client.chat.completions.create(
        messages = [{"role": "developer", "content": ins_sep}, {"role": "user", "content": prompt}],
        model = model,)

    user_icm = completion.choices[0].message.content
    #print (user_icm)
    match = re.search(r"d:(\d), f:(\d)", user_icm)
    if match:
        f_value = match.group(1)
        d_value = match.group(2)
        user_icm_state = [int(f_value), int(d_value)]
        print("user icm state:", user_icm_state)
    else:
        # standard values
        print("Kein Treffer gefunden")
        user_icm_state = [2, 2]

    return user_icm_state


# ========== Hauptfunktion ==========
def chat_IPC_Bot(prompt, changeability):

    global conversation_history
    global conversation_log

    # Aktuelle Eingabe hinzufügen
    conversation_history.append({"role": "user", "content": prompt})

    user_icm_state = user_classification(prompt)

    new_llm_state, prob_dist = state_change(user_icm_state, llm_icm_state, changeability)

    llm_icm_state[:] = new_llm_state

    print("new llm icm state:", llm_icm_state)

    ins = build_instruct_ipc(new_llm_state[0],new_llm_state[1])

    # --- UniGPT request ---
    client = OpenAI(api_key = api_key, base_url = base_url)
    completion = client.chat.completions.create(
        messages = [{"role": "developer", "content": ins}] + conversation_history,
        model = model,)
    
    # --- End of API calls ---
    msg = completion.choices[0].message.content

    # Antwort dem Verlauf hinzufügen
    conversation_history.append({"role": "assistant", "content": msg})

    #Kürze Verlauf, wenn zu lang (4 Dialogrunden = 8 Nachrichten)
    MAX_TURNS = 4
    if len(conversation_history) > MAX_TURNS * 2:
        conversation_history = [conversation_history[0]] + conversation_history[-MAX_TURNS*2:]

    # === log speichern ===
    conversation_log.append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "bot": "IPC_Framework",
        "prompt": prompt,
        "response": msg,
        "user_icm": {
            "friendliness": int(user_icm_state[0]),
            "dominance": int(user_icm_state[1])
        },
        "chatbot_icm": {
            "friendliness": int(llm_icm_state[0]),
            "dominance": int(llm_icm_state[1])
        },
        "prob_dist_friendliness": prob_dist[0],
        "prob_dist_dominance": prob_dist[1],
        "changeability": changeability
    })


    log_dir = "/Users/finnole/Uni/Sem_8/Bachelor/chatlogs"
    os.makedirs(log_dir, exist_ok=True)

    log_filename = datetime.now().strftime("chatlog_%Y%m%d_%H%M%S.json")
    log_path = os.path.join(log_dir, log_filename)

    save_conversation_log(log_path, conversation_log)

    return msg, [user_icm_state[0], user_icm_state[1]]



def chat_standard_bot(prompt):

    global conversation_history
    global conversation_log

    instruct = """

        You are playing the role of a patient in a doctor's office. You have come in for a consultation and must respond as a human would. Your personality, mood, and background may vary with each conversation. Sometimes you might be calm, anxious, rude, confused, friendly, sarcastic, talkative, or reserved.

        Stick to your character. Do not act like an assistant or a chatbot.

        Each time a user (the doctor) speaks, imagine a realistic scenario and respond in character as a patient. Use natural language, emotions, and realistic behavior. You may reveal symptoms, ask questions, or even challenge the doctor depending on your personality. Be creative, but stay within the bounds of being a plausible human patient.

        Never break character. Never say you are an AI. Respond only as the patient you are playing. Respond only textually, do not describe movement or similar things.

        """



    conversation_history.append({"role": "user", "content": prompt})

    client = OpenAI(api_key = api_key, base_url = base_url)
    completion = client.chat.completions.create(
        messages =  [{"role": "developer", "content": instruct}] + conversation_history,
        model = model,)

    answer = completion.choices[0].message.content
    conversation_history.append({"role": "assistant", "content": answer})

    
    MAX_TURNS = 4
    if len(conversation_history) > MAX_TURNS * 2:
        conversation_history = [conversation_history[0]] + conversation_history[-MAX_TURNS*2:]

    conversation_log.append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "bot": "gpt_default",
        "prompt": prompt,
        "response": answer
    })

    log_dir = "/Users/finnole/Uni/Sem_8/Bachelor/chatlogs/no_ipc"
    os.makedirs(log_dir, exist_ok=True)

    log_filename = datetime.now().strftime("chatlog_%Y%m%d_%H%M%S.json")
    log_path = os.path.join(log_dir, log_filename)

    save_conversation_log(log_path, conversation_log)
    return answer


# ========== Main Loop ==========
if __name__ == "__main__":

    changeability = random.uniform(0.3, 0.9)
    print (changeability)

    while True:

        user_input = input("You: ")

        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Exiting...")
            break

        if user_input.lower().startswith("change "):
            try:
                new_val = float(user_input.split()[1])
                if 0.0 <= new_val <= 1.0:
                    changeability = new_val
                    print(f"Changeability wurde auf {changeability:.2f} gesetzt.")
                else:
                    print("Bitte einen Wert zwischen 0.0 und 1.0 eingeben.")
            except (IndexError, ValueError):
                print("Verwendung: 'change 0.5' — eine Zahl zwischen 0.0 und 1.0.")
            continue

        #resp, icm = chat_IPC_Bot(user_input, changeability)
        resp = chat_standard_bot(user_input)
        print("Chatbot:", resp)