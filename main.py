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

f = ["very friendly", "slightly friendly", "neutrally friendly", "slightly hostile", "very hostile"]
d = ["very submissive","slightly submissive","neutrally submissive","slightly dominant","very dominant"]

# startwert, noch zufällig machen
llm_icm_state = [2, 2]

# ======= instructions aus txt laden =======
def load_dynamic_instructions(llm_icm_state):
    with open("instructions.txt", "r", encoding="utf-8") as file:
        ins = file.read()

    # Hole Labels
    f_label = f[llm_icm_state[0]]
    d_label = d[llm_icm_state[1]]

    # Ersetze die Zeile dynamisch (RegEx für mehr Robustheit)
    ins = re.sub(
        r"\* When talking you are .*?, according to the interpersonal circumplex model",
        f"* When talking you are {f_label} and {d_label}, according to the interpersonal circumplex model",
        ins
    )
    return ins


# ========== Hauptfunktion ==========
def chat_input(prompt, changeability):

    global conversation_history
    global conversation_log

    ins = load_dynamic_instructions(llm_icm_state)

    # Initialisiere Verlauf bei erstem Aufruf
    if not conversation_history:
        conversation_history.append({"role": "developer", "content": ins})

    # Aktuelle Eingabe hinzufügen
    conversation_history.append({"role": "user", "content": prompt})

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
        messages = conversation_history,
        model = model,)
    
    # --- End of API calls ---
    
    msg = completion.choices[0].message.content

    # Antwort dem Verlauf hinzufügen
    conversation_history.append({"role": "assistant", "content": msg})

    #Kürze Verlauf, wenn zu lang (4 Dialogrunden = 8 Nachrichten)
    MAX_TURNS = 4
    if len(conversation_history) > MAX_TURNS * 2 + 1:  # +1 für developer prompt
        conversation_history = [conversation_history[0]] + conversation_history[-MAX_TURNS*2:]


    # ========== ICM teil extrahieren ==========

    #teilweise unvollständiger exit tag
    match = match = re.search(r"<icm>\s*({.*?})\s*</ic(?:m)?>", msg, re.DOTALL)
    if not match:
        print("Kein <icm>{...}</icm> Block im LLM-Output gefunden. Ausgabe war:")
        print(msg)
        return msg.strip(), None 

    #teilweise kein richtiges json
    raw_json_str = match.group(1)
    try:
        icm_data = json.loads(raw_json_str)
    except json.JSONDecodeError:
        try:
            unescaped = json.loads(f'"{raw_json_str}"')
            icm_data = json.loads(unescaped)
        except Exception as e2:
            print("Konnte ICM nicht lesen:", e2)
            print("Rohdaten:", raw_json_str)
            return msg.strip(), None


    user_icm_state = [icm_data['friendliness'], icm_data['dominance']]

    msg_clean = re.sub(r"<icm>.*?</ic(?:m)", "", msg).strip()


    new_llm_state = state_change(user_icm_state, llm_icm_state, changeability)
    print(f"ICM-Update: User war (f={user_icm_state[0]}, d={user_icm_state[1]}), LLM war (f={llm_icm_state[0]}, d={llm_icm_state[1]}), jetzt (f={new_llm_state[0]}, d={new_llm_state[1]})")

    llm_icm_state[:] = new_llm_state

    # === Speichern mit ID und Timestamp ===
    conversation_log.append({
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "prompt": prompt,
        "response": msg_clean,
        "user_icm": icm_data,
        "chatbot_icm": {
            "friendliness": llm_icm_state[0],
            "dominance": llm_icm_state[1]
        },
        "changeability": changeability
    })
    save_conversation_log(log_path, conversation_log)

    return msg_clean, icm_data


# ========== ICM Update-Logik ==========
def state_change(user_state, current_llm, changeability):
    """
    changeability ∈ [0.0, 1.0]: 0 = unbeweglich, 1 = maximal reaktiv
    user_state: [friendliness, dominance] ∈ [0..4]
    current_llm: [friendliness, dominance] ∈ [0..4]
    returns: [new_friendliness, new_dominance]
    """
    
    def get_next_axis_value(user_val, current_val):
        possible_values = range(5)
        options = {}

        for value in possible_values:
            diff_to_user = abs(value - user_val)

            # Grundwahrscheinlichkeit basierend auf Nähe zum user
            base_prob = np.exp(-diff_to_user)

            # changeability:
            # - bei niedriger changeability bleibt man beim current_val
            # - bei hoher changeability bewegt man sich eher zum user_val
            stay_bias = (value == current_val) * (1 - changeability)  # stark bei niedriger changeability
            move_bias = changeability * (1 / (1 + diff_to_user))  # Bewegungskomponente
            tendency = (stay_bias + move_bias)
            prob = base_prob * tendency
            print (value , " hat base wahrscheinlichkeit: ", base_prob, " und tendenz: ", tendency, " also prob: ", prob)
            options[value] = prob

        # Normierung
        total = sum(options.values())
        for k in options:
            options[k] /= total

        values, weights = zip(*options.items())
        return random.choices(values, weights=weights, k=1)[0]

    # Für beide Dimensionen berechnen
    new_friendliness = get_next_axis_value(user_state[0], current_llm[0])
    new_dominance = get_next_axis_value(user_state[1], current_llm[1])
    print(new_friendliness, new_dominance)

    return [new_friendliness, new_dominance]


def optional_doctor_bot(chat_prompt):
    client = OpenAI(api_key=api_key, base_url=base_url)
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a doctor. You are chatting with a patient. He will tell you some of his symptoms but you have to figure it out."},
            {"role": "user", "content": chat_prompt}
        ],
        model=model,
    )
    msg = completion.choices[0].message.content
    return msg



# ========== Main Loop ==========
if __name__ == "__main__":

    changeability = random.uniform(0.3, 0.9)
    print (changeability)
    doctor_patient_input = None  # speichert, was Patient dem Doctor sagt

    state_change([2,2],[2,2], 1)

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

        if user_input.lower() == "doctor":
            # Wenn doctor eingegeben wird, dann DoctorBot antwortet auf Patienteneingabe
            # Falls keine Patienteneingabe vorhanden, nimm letzte Antwort vom Hauptbot
            if doctor_patient_input is None:
                # Suche letzte Antwort des Hauptbots
                last_bot_msg = None
                for msg in reversed(conversation_history):
                    if msg["role"] == "assistant":
                        last_bot_msg = msg["content"]
                        break
                if last_bot_msg is None:
                    print("Keine letzte Antwort des Hauptbots gefunden.")
                    continue
                doctor_patient_input = last_bot_msg

            # DoctorBot antwortet
            doctor_resp = optional_doctor_bot(doctor_patient_input)
            print("DoctorBot:", doctor_resp)

            # Setze Patienteneingabe als Antwort des DoctorBots (für nächstes Mal)
            doctor_patient_input = doctor_resp

            # Die Eingabe "doctor" wird **nicht** an den Hauptchat gesendet,
            # also verschwindet sie quasi und wird nicht geloggt.
            continue

        # Wenn normaler Input, dann normale Hauptbot Konversation
        doctor_patient_input = None  # reset Patienteneingabe, da User wieder manuell spricht
        resp, icm = chat_input(user_input, changeability)
        print("Chatbot:", resp)