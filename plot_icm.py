import json
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import os
import numpy as np
from datetime import datetime

# Pfad zur Datei (optional anpassen)
log_file_path = "/Users/finnole/Uni/Sem_8/Bachelor/chatlogs/chatlog_20250618_143727.json"  # <-- Aktuelle Datei eintragen

# Laden
with open(log_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# ---- Bot-Werte vorbereiten ----
bot_coords = []
for entry in data:
    f = entry["chatbot_icm"]["friendliness"] - 2
    d = entry["chatbot_icm"]["dominance"] - 2
    bot_coords.append((d, f))  # x = Dominanz, y = Freundlichkeit

bot_coords = np.array(bot_coords)
xs, ys = bot_coords[:, 0], bot_coords[:, 1]

# ==== 🎨 Farbverlauf über Zeit ====
colors = cm.viridis(np.linspace(0, 1, len(xs)))

# ==== 📈 Plot Setup ====
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(-2.5, 2.5)
ax.set_ylim(-2.5, 2.5)
ax.axhline(0, color='gray', linewidth=0.5)
ax.axvline(0, color='gray', linewidth=0.5)
ax.grid(True, linestyle='--', alpha=0.3)
ax.set_xlabel("Dominanz")
ax.set_ylabel("Freundlichkeit")
ax.set_title("ICM-Verlauf des Chatbots im 2D-Raum")

# ==== ➡️ Verlaufspfeile zeichnen ====
for i in range(len(xs) - 1):
    ax.arrow(xs[i], ys[i], xs[i+1] - xs[i], ys[i+1] - ys[i],
             head_width=0.1, length_includes_head=True,
             color=colors[i], alpha=0.8)

# ==== 🟢🔴 Start- und Endpunkte markieren ====
ax.plot(xs[0], ys[0], marker='o', color='green', label="Start")
ax.plot(xs[-1], ys[-1], marker='s', color='red', label="Ende")

# ==== 🗺️ Farblegende hinzufügen ====
sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=0, vmax=len(xs)-1))
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Dialogrunde (zeitlicher Verlauf)")

ax.legend()
plt.tight_layout()

# ==== 💾 Speichern am gewünschten Ort ====
output_dir = "/Users/finnole/Uni/Sem_8/Bachelor/chatlogs/"  # kannst du beliebig anpassen
os.makedirs(output_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_dir, f"icm_bot_path_{timestamp}.png")
plt.savefig(output_path, dpi=300)
plt.show()

print(f"Diagramm erfolgreich gespeichert unter: {output_path}")
