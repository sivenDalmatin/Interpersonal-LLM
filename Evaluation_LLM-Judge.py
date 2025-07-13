import json
import os
import openai
from datetime import datetime
from dotenv import load_dotenv

load_dotenv() 
openai.api_key = os.getenv("CHAT_API_KEY")

# Evaluation Prompt
EVALUATION_PROMPT = """
You are an impartial and analytical judge tasked with evaluating the quality of a full dialogue between a doctor and a simulated AI patient. Focus only on the patient's responses (i.e., every second turn in the conversation).

Your evaluation should consider three criteria:

1. Authenticity – Does the patient’s behavior resemble that of a real human in a clinical setting?
2. Emotional Appropriateness – Are emotional responses appropriate for the unfolding situation and consistent with a believable personality?
3. Behavioral Coherence – Does the patient's character stay consistent in behavior, tone, and emotional style throughout the conversation, based on their apparent personality (such as dominant/passive or warm/cold)?

Please provide a brief explanation (2–4 sentences) and then score the performance on a 1–10 scale, following this exact format:

Authenticity: [[Rating]]
Emotional Appropriateness: [[Rating]]
Behavioral Coherence: [[Rating]]


Do not include any other text after the ratings.
"""

def evaluate_conversation(conversation_turns):
    dialogue = ""
    for turn in conversation_turns:
        prompt = turn.get("prompt", "").strip()
        response = turn.get("response", "").strip()
        if prompt:
            dialogue += f"Doctor: {prompt}\n"
        if response:
            dialogue += f"Patient: {response}\n"

    messages = [
        {"role": "system", "content": EVALUATION_PROMPT},
        {"role": "user", "content": dialogue.strip()}
    ]

    try:
        resp = openai.responses.create(
            #model="gpt-3.5-turbo",
            model = "gpt-4.0",
            input = messages
        )
        content = resp.output[0].content[0].text
    except Exception as e:
        print("❌ OpenAI API call failed:", e)
        return None

    lines = content.splitlines()
    ratings = {}
    for line in lines:
        if line.startswith("Authenticity:"):
            ratings["authenticity"] = int(line.split(":")[1].strip().strip("[]"))
        elif line.startswith("Emotional Appropriateness:"):
            ratings["emotional_appropriateness"] = int(line.split(":")[1].strip().strip("[]"))
        elif line.startswith("Behavioral Coherence:"):
            ratings["behavioral_coherence"] = int(line.split(":")[1].strip().strip("[]"))

    justification = "\n".join([
        line for line in lines
        if not any(line.startswith(label) for label in ["Authenticity:", "Emotional Appropriateness:", "Behavioral Coherence:"])
    ])

    evaluation = {
        "authenticity": ratings.get("authenticity"),
        "emotional_appropriateness": ratings.get("emotional_appropriateness"),
        "behavioral_coherence": ratings.get("behavioral_coherence"),
        "justification": justification,
        "evaluator": "llm judge",
        "timestamp": datetime.utcnow().isoformat()
    }

    return evaluation

def process_file(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, list):
        dialogue = raw
        metadata = {"dialogue": dialogue, "evaluation": []}
    elif isinstance(raw, dict):
        metadata = raw
        dialogue = metadata.get("dialogue", [])
        if "evaluation" not in metadata:
            metadata["evaluation"] = []
    else:
        print(f"❌ Skipping invalid file: {json_path}")
        return

    evaluation = evaluate_conversation(dialogue)
    if evaluation:
        metadata["evaluation"].append(evaluation)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Evaluation added to: {json_path}")
    else:
        print(f"❌ Evaluation failed for: {json_path}")

def process_folder(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
    if not files:
        print("⚠️ No JSON files found.")
        return

    for file_name in files:
        full_path = os.path.join(folder_path, file_name)
        print(f"📄 Processing: {file_name}")
        process_file(full_path)

# Example usage
if __name__ == "__main__":
    folder_path = "/Users/finnole/Uni/Sem_8/Bachelor/chatlogs"
    #process_folder(folder_path)
    process_file()
