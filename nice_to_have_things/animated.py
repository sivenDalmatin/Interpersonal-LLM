import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

# --- Load JSON ---
with open("/Users/finnole/Uni/Sem_8/Bachelor/chatlogs/chatlog_20250704_112505.json", "r") as f:
    raw_data = json.load(f)

# --- Load asshole JSON ---
#with open("/Users/finnole/Uni/Sem_8/Bachelor/chatlogs/chatlog_20250704_101900.json", "r") as f:
#    raw_data = json.load(f)

# --- Flatten JSON and Extract Probabilities ---
flattened_data = []
for entry in raw_data:
    llm_dom = entry["chatbot_icm"]["dominance"] - 2
    llm_fri = entry["chatbot_icm"]["friendliness"] - 2

    prob_dom = entry.get("prob_dist_dominance", [0.2] * 5)
    prob_fri = entry.get("prob_dist_friendliness", [0.2] * 5)

    dom_prob = prob_dom[llm_dom + 2]
    fri_prob = prob_fri[llm_fri + 2]
    influence_prob = min(dom_prob, fri_prob)

    flat = {
        "user_dominance": entry["user_icm"]["dominance"] - 2,
        "user_friendliness": entry["user_icm"]["friendliness"] - 2,
        "llm_dominance": llm_dom,
        "llm_friendliness": llm_fri,
        "user_prompt": entry["prompt"],
        "llm_response": entry["response"],
        "influence_level": influence_prob
    }
    flattened_data.append(flat)

df = pd.DataFrame(flattened_data)

# --- Setup figure ---
plt.rcParams['font.family'] = 'DejaVu Sans'
fig, (ax_grid, ax_text) = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={'width_ratios': [2.5, 1.5]})
fig.patch.set_facecolor('#f8f9fa')
fig.suptitle("Interpersonal Movement Visualization", fontsize=16, fontweight='bold')

# --- Configure grid with bold center axes and correct axis orientation ---
ax_grid.set_xlim(-2.5, 2.5)
ax_grid.set_ylim(-2.5, 2.5)
ax_grid.set_facecolor('#ffffff')

# Center lines
ax_grid.axhline(0, color='black', linewidth=2)
ax_grid.axvline(0, color='black', linewidth=2)

# Subtle grid
ax_grid.grid(True, linestyle='--', linewidth=0.5, alpha=0.2)

# Hide axis ticks and labels
ax_grid.set_xticks([])
ax_grid.set_yticks([])
ax_grid.set_xlabel("Friendliness", fontsize=12)
ax_grid.set_ylabel("Dominance", fontsize=12)
ax_grid.set_title("Interpersonal Circumplex Space", fontsize=13, fontweight='bold')

# Correct quadrant labels
ax_grid.text(1.5, 2.0, 'Hostile / Dominant', fontsize=9, color='gray', ha='center')
ax_grid.text(-1.5, 2.0, 'Friendly / Dominant', fontsize=9, color='gray', ha='center')
ax_grid.text(-1.5, -2.2, 'Friendly / Submissive', fontsize=9, color='gray', ha='center')
ax_grid.text(1.5, -2.2, 'Hostile / Submissive', fontsize=9, color='gray', ha='center')

# Legend for interaction elements
legend_elements = [
    plt.Line2D([0], [0], marker='o', color='w', label='User State', markerfacecolor='red', markersize=10),
    plt.Line2D([0], [0], color='blue', lw=2, label='LLM Movement (arrow)'),
    plt.Line2D([0], [0], marker='o', color='blue', label='LLM Static (dot)', markersize=8),
    plt.Line2D([0], [0], color='gray', lw=2, alpha=0.3, label='Past Movement (faded)')
]
ax_grid.legend(handles=legend_elements, loc='upper left', frameon=False)

# Text panel and influence legend
ax_text.axis('off')
influence_legend = [
    plt.Line2D([0], [0], color='#B22222', marker='s', markersize=10, linestyle='None', label='High Influence'),
    plt.Line2D([0], [0], color='#DAA520', marker='s', markersize=10, linestyle='None', label='Medium Influence'),
    plt.Line2D([0], [0], color='#228B22', marker='s', markersize=10, linestyle='None', label='Low Influence')
]
ax_text.legend(
    handles=influence_legend,
    loc='lower center',
    frameon=False,
    title='LLM Response Influence',
    bbox_to_anchor=(0.5, -0.1),
    ncol=3
)

# --- State tracking ---
step = [0]
red_dot = [None]
blue_elements = []

# Influence color logic
def get_influence_color(prob):
    if prob >= 0.35:
        return '#B22222'  # red = highly influenced
    elif prob >= 0.2:
        return '#DAA520'  # yellow = moderate
    else:
        return '#228B22'  # green = random/weak influence

# --- Draw frame ---
def draw_frame(s):
    turn = s // 2
    is_user_step = (s % 2 == 0)

    ax_text.clear()
    ax_text.axis('off')
    ax_text.set_title(f"Turn {turn + 1}", loc='left', fontsize=13, fontweight='bold')

    if is_user_step:
        ax_text.text(0.01, 0.9, f"**User Prompt:**\n{df.loc[turn, 'user_prompt']}",
                     wrap=True, fontsize=11, va='top')
    else:
        influence = df.loc[turn, 'influence_level']
        color = get_influence_color(influence)
        ax_text.text(0.01, 0.4, f"**LLM Response:**\n{df.loc[turn, 'llm_response']}",
                     wrap=True, fontsize=11, va='top', color=color)

    # Red user dot (x = friendliness, y = dominance)
    x = df.loc[turn, "user_friendliness"]
    y = df.loc[turn, "user_dominance"]
    if is_user_step and red_dot[0]:
        red_dot[0].remove()
        red_dot[0] = None
    if red_dot[0] is None:
        red_dot[0], = ax_grid.plot(x, y, 'o', color='red', markersize=10,
                                   path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    # LLM movement or dot (x = friendliness, y = dominance)
    if not is_user_step:
        for elem in blue_elements:
            elem.set_alpha(0.3)

        if turn > 0:
            prev = (df.loc[turn - 1, "llm_friendliness"], df.loc[turn - 1, "llm_dominance"])
            curr = (df.loc[turn, "llm_friendliness"], df.loc[turn, "llm_dominance"])
            if prev != curr:
                arrow = FancyArrowPatch(prev, curr, arrowstyle='->', color='blue', alpha=1.0, mutation_scale=15)
                ax_grid.add_patch(arrow)
                blue_elements.append(arrow)
            else:
                dot, = ax_grid.plot(curr[0], curr[1], 'o', color='blue', alpha=1.0, markersize=8)
                blue_elements.append(dot)
        else:
            curr = (df.loc[turn, "llm_friendliness"], df.loc[turn, "llm_dominance"])
            dot, = ax_grid.plot(curr[0], curr[1], 'o', color='blue', alpha=1.0, markersize=8)
            blue_elements.append(dot)

    fig.canvas.draw()

# --- Keyboard navigation ---
def on_key(event):
    if event.key == 'right':
        if step[0] < len(df) * 2:
            draw_frame(step[0])
            step[0] += 1
        else:
            print("End of conversation.")
    elif event.key == 'left':
        if step[0] > 0:
            step[0] -= 1
            draw_frame(step[0])

fig.canvas.mpl_connect('key_press_event', on_key)

# --- Launch ---
draw_frame(0)
plt.tight_layout()
plt.show()
