

import matplotlib.pyplot as plt

import numpy as np

import json

# Path to your chatlog file
json_path = "/Users/finnole/Uni/Sem_8/Bachelor/chatlogs/chatlog_20250618_143727.json"  # <- change this to your actual file path
with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

def verlauf():
    # Extract values for plotting
    chatbot_f = [entry["chatbot_icm"]["friendliness"] for entry in data]
    chatbot_d = [entry["chatbot_icm"]["dominance"] for entry in data]
    user_f = [entry["user_icm"]["friendliness"] for entry in data]
    user_d = [entry["user_icm"]["dominance"] for entry in data]
    x = list(range(len(data)))

    # === Friendliness Plot ===
    plt.figure(figsize=(10, 4))
    plt.plot(x, chatbot_f, label="Chatbot Friendliness", marker="o")
    plt.scatter(x, user_f, color="red", label="User Friendliness", zorder=5)
    plt.title("Friendliness Over Time")
    plt.xlabel("Turn")
    plt.ylabel("Friendliness (0–4)")
    plt.ylim(-0.5, 4.5)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/Users/finnole/Uni/Sem_8/Bachelor/graphs/friendliness_plot_for_chatlog_20250618_143727.png")  # Save as PNG
    plt.show()

    # === Dominance Plot ===
    plt.figure(figsize=(10, 4))
    plt.plot(x, chatbot_d, label="Chatbot Dominance", marker="o")
    plt.scatter(x, user_d, color="red", label="User Dominance", zorder=5)
    plt.title("Dominance Over Time")
    plt.xlabel("Turn")
    plt.ylabel("Dominance (0–4)")
    plt.ylim(-0.5, 4.5)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("/Users/finnole/Uni/Sem_8/Bachelor/graphs/dominance_plot_for_chatlog_20250618_143727.png")  # Save as PNG
    plt.show()




def plot_probability_distributions(data, save_dir="/Users/finnole/Uni/Sem_8/Bachelor/graphs/probcharts_for_chatlog_20250618_143727"):
    import os
    os.makedirs(save_dir, exist_ok=True)

    for i, entry in enumerate(data):
        x = np.arange(5)

        # --- Friendliness ---
        plt.figure(figsize=(6, 3))
        plt.bar(x, entry["prob_dist_friendliness"], color="blue")
        plt.xticks(x, [0, 1, 2, 3, 4])
        plt.title(f"Turn {i} - Friendliness Distribution")
        plt.xlabel("Friendliness Value")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/friendliness_dist_turn_{i}.png")
        plt.close()

        # --- Dominance ---
        plt.figure(figsize=(6, 3))
        plt.bar(x, entry["prob_dist_dominance"], color="green")
        plt.xticks(x, [0, 1, 2, 3, 4])
        plt.title(f"Turn {i} - Dominance Distribution")
        plt.xlabel("Dominance Value")
        plt.ylabel("Probability")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/dominance_dist_turn_{i}.png")
        plt.close()

#plot_probability_distributions(data)
verlauf()
