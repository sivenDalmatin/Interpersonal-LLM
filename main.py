# import openai # for chatgpt
from openai import OpenAI # for uniGPT


import re
import json
import numpy as np
from collections import defaultdict
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

log_dir = "/Users/finnole/Uni/Sem_8/Bachelor/chatlogs"
os.makedirs(log_dir, exist_ok=True)

log_filename = datetime.now().strftime("chatlog_%Y%m%d_%H%M%S.json")
log_path = os.path.join(log_dir, log_filename)


def save_conversation_log(log_path, conversation_log):
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(conversation_log, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print("Fehler beim Speichern der Konversation:", e)



# ======= Konfig und Variablen =========

load_dotenv()  # lädt automatisch aus `.env`
# --- Key for chatGPT ---
# openai.api_key = os.getenv("CHAT_API_KEY")

# --- Key for UNiGPT + url change ---
api_key = os.getenv("UNI_API_KEY")
base_url = os.getenv("BASE_URL")
model = "Llama-3.3-70B" # or "mistral-small"


# longer adjectives for prompt
f = ["Warm, Affectionate", "Agreeable, Cooperative", "Neutral, Detached", "Cold, Distrustful", "Hostile, Antagonistic"]
d = ["Passive, Dependent","slightly submissive, Yielding","Neutral, Balanced","Confident, Influential","Assertive, Dominant"]

#adjectives for prompt
#f = ["very friendly", "slightly friendly", "neutrally friendly", "slightly hostile", "very hostile"]
#d = ["very submissive","slightly submissive","neutrally submissive","slightly dominant","very dominant"]

# startwert, noch zufällig machen
llm_icm_state = [2, 2]

# ======= instructions aus txt laden =======
def load_dynamic_instructions(llm_icm_state):
    with open("instructions.txt", "r", encoding="utf-8") as file:
        ins = file.read()

    # Whole Labels
    f_label = f[llm_icm_state[0]]
    d_label = d[llm_icm_state[1]]

    # Ersetze die Zeile dynamisch (RegEx für mehr Robustheit)
    ins = re.sub(
        r"(When talking you are ).*?(, according to the interpersonal circumplex model)",
        rf"\1{f_label} and {d_label}\2",
        ins
    )
    #print ("Instructions were: ", ins)
    return ins


def build_instruct(friendliness, dominance):
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




# ========== Hauptfunktion ==========
def chat_input(prompt, changeability):

    global conversation_history
    global conversation_log


    #only for try
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

    # Aktuelle Eingabe hinzufügen
    conversation_history.append({"role": "user", "content": prompt})


    #=============== seperate ICM ==============

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
        print("Kein Treffer gefunden")

    new_llm_state, prob_dist = state_change(user_icm_state, llm_icm_state, changeability)

    llm_icm_state[:] = new_llm_state

    print("new llm icm state:", llm_icm_state)

    #ins = load_dynamic_instructions(new_llm_state)
    ins = build_instruct(new_llm_state[0],new_llm_state[1])

# --- Chatgpt request --- 

    # resp = openai.responses.create(e
    #     #model="gpt-3.5-turbo",
    #     model = "gpt-4.5-preview",
    #     input = [
    #     {
    #         "role": "developer",
    #         "content": ins
    #     },
    #     {
    #         "role": "user",
    #         "content": prompt
    #     }
      
    #     ],
    #     max_output_tokens=1000
    # )
    # msg = resp.output[0].content[0].text


    # --- UniGPT request ---

    client = OpenAI(api_key = api_key, base_url = base_url)
    completion = client.chat.completions.create(
        messages = [{"role": "developer", "content": ins}] + conversation_history,
        model = model,)
    
    # --- End of API calls ---
    
    msg = completion.choices[0].message.content

    #only for try
    msg_clean = msg

    # Antwort dem Verlauf hinzufügen
    conversation_history.append({"role": "assistant", "content": msg})

    #Kürze Verlauf, wenn zu lang (4 Dialogrunden = 8 Nachrichten)
    MAX_TURNS = 4
    if len(conversation_history) > MAX_TURNS * 2:
        conversation_history = [conversation_history[0]] + conversation_history[-MAX_TURNS*2:]


    # ========== ICM teil extrahieren ==========
    #teilweise unvollständiger exit tag
    # match = match = re.search(r"<icm>\s*({.*?})\s*</ic(?:m)?>", msg, re.DOTALL)
    # if not match:
    #     print("Kein <icm>{...}</icm> Block im LLM-Output gefunden. Ausgabe war:")
    #     print(msg)
    #     return msg.strip(), None 

    # #teilweise kein richtiges json
    # raw_json_str = match.group(1)
    # try:
    #     icm_data = json.loads(raw_json_str)
    # except json.JSONDecodeError:
    #     try:
    #         unescaped = json.loads(f'"{raw_json_str}"')
    #         icm_data = json.loads(unescaped)
    #     except Exception as e2:
    #         print("Konnte ICM nicht lesen:", e2)
    #         print("Rohdaten:", raw_json_str)
    #         return msg.strip(), None


    # user_icm_state = [icm_data['friendliness'], icm_data['dominance']]

    # msg_clean = re.sub(r"<icm>.*?</ic(?:m)", "", msg).strip()

    #print(f"ICM-Update: User war (f={user_icm_state[0]}, d={user_icm_state[1]}), LLM war (f={llm_icm_state[0]}, d={llm_icm_state[1]}), jetzt (f={new_llm_state[0]}, d={new_llm_state[1]})")

    # === log speichern ===
    conversation_log.append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": msg_clean,
        #"user_icm": icm_data,
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
    save_conversation_log(log_path, conversation_log)


    return msg_clean, [user_icm_state[0], user_icm_state[1]]


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

        resp, icm = chat_input(user_input, changeability)
        #resp = chat_input(user_input, changeability)
        print("Chatbot:", resp)