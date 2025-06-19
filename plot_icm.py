import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import numpy as np
from datetime import datetime

# Pfad zur Datei (optional anpassen)
log_file_path = "/Users/finnole/Uni/Sem_8/Bachelor/chatlogs/chatlog_20250619_155407.json"  # <-- Aktuelle Datei eintragen

# Laden
def load_conversation_log(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

log_data = load_conversation_log(log_file_path)


def get_axis_probability_distribution(user_val, current_val, changeability):
    possible_values = range(5)
    options = {}

    for value in possible_values:
        diff_to_user = abs(value - user_val)

        # Nähe zum user_val — exponentiell abnehmend
        base_prob = np.exp(-diff_to_user)

        # Mischung je nach changeability
        if value == current_val:
            tendency = (1 - changeability)
        else:
            tendency = changeability * (1 / (1 + diff_to_user))

        prob = base_prob * tendency
        options[value] = prob

    # Normierung
    total = sum(options.values())
    for k in options:
        options[k] /= total

    return options

def compute_2d_distribution(user_state, current_state, changeability):
    print (user_state)
    dist_f = get_axis_probability_distribution(user_state["friendliness"], current_state["dominance"], changeability)
    dist_d = get_axis_probability_distribution(user_state["friendliness"], current_state["dominance"], changeability)
    matrix = np.zeros((5, 5))
    for f in range(5):
        for d in range(5):
            matrix[d, f] = dist_f[f] * dist_d[d]
    return matrix

def plot_single_turn_heatmap(matrix, user_icm=None, chatbot_icm=None, changeability=None, title="Wahrscheinlichkeitsverteilung"):
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix, cmap="Blues", origin="lower", vmin=0, vmax=1)

    # Prozentwerte in der Heatmap anzeigen
    for y in range(5):
        for x in range(5):
            percent_val = matrix[y, x] * 100
            plt.text(x, y, f"{percent_val:.1f}%", ha="center", va="center", color="black")

    plt.title(title)
    plt.xlabel("Friendliness")
    plt.ylabel("Dominance")
    plt.xticks(range(5))
    plt.yticks(range(5))

    # Farbskala mit Prozentbeschriftung
    cbar = plt.colorbar()
    cbar.set_label("Wahrscheinlichkeit (%)")
    cbar.set_ticks(np.linspace(0, 1, 6))  # 0%, 20%, ..., 100%
    cbar.set_ticklabels([f"{int(t * 100)}%" for t in np.linspace(0, 1, 6)])

    # Zusatzinfos als Text unter dem Plot
    info_lines = []
    if user_icm:
        info_lines.append(f"User ICM: Friendliness = {user_icm['friendliness']}, Dominance = {user_icm['dominance']}")
    if chatbot_icm:
        info_lines.append(f"Chatbot ICM: Friendliness = {chatbot_icm['friendliness']}, Dominance = {chatbot_icm['dominance']}")
    if changeability is not None:
        info_lines.append(f"Changeability = {changeability:.2f}")
    
    info_text = "\n".join(info_lines)
    # Text unterhalb des Plots anzeigen
    plt.gcf().text(0.02, 0.01, info_text, fontsize=10, va='bottom', ha='left')

    # Tight layout mit Platz unten
    plt.tight_layout(rect=[0, 0.07, 1, 1])
    plt.show()



def plot_fading_heatmap_sequence(log_data, alpha_start=1.0, alpha_decay=0.15):
    plt.figure(figsize=(6, 5))
    base_matrix = np.zeros((5, 5))
    
    for i, entry in enumerate(log_data):
        user_icm = entry["user_icm"]
        chatbot_icm = entry["chatbot_icm"]
        changeability = entry["changeability"]

        matrix = compute_2d_distribution(user_icm, chatbot_icm, changeability)

        alpha = max(alpha_start - i * alpha_decay, 0.1)  # nie ganz unsichtbar
        plt.imshow(matrix, cmap="Blues", origin="lower", alpha=alpha, vmin=0, vmax=1)

    plt.title("Verlauf der Wahrscheinlichkeitsverteilung (ältere verblasst)")
    plt.xlabel("Friendliness")
    plt.ylabel("Dominance")
    plt.xticks(range(5))
    plt.yticks(range(5))
    plt.colorbar(label="Wahrscheinlichkeit")
    plt.tight_layout()
    plt.show()


# Letzter Abschnitt als Einzel-Heatmap
last = log_data[-1]
matrix = compute_2d_distribution(last["user_icm"], last["chatbot_icm"], last["changeability"])
plot_single_turn_heatmap(
    matrix,
    user_icm=last["user_icm"],
    chatbot_icm=last["chatbot_icm"],
    changeability=last["changeability"],
    title="Letzter Dialogschritt"
)