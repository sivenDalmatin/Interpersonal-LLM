from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

def personality_detection(text, threshold=0.05, endpoint=1.0):
    token = os.getenv("TOKEN_CLASS")
    tokenizer = AutoTokenizer.from_pretrained("Nasserelsaman/microsoft-finetuned-personality", token=token)
    model = AutoModelForSequenceClassification.from_pretrained("Nasserelsaman/microsoft-finetuned-personality", token=token)

    inputs = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = torch.sigmoid(logits)  # shape: [1, 5]

    # Clamp to threshold and endpoint
    probabilities = torch.clamp(probabilities, min=threshold, max=endpoint)

    label_names = ['Agreeableness', 'Conscientiousness', 'Extraversion', 'Neuroticism', 'Openness']
    result = {label: float(probabilities[0, i]) for i, label in enumerate(label_names)}

    return result


def compute_ipc_from_big5(b5_scores):
    extr = b5_scores["Extraversion"]
    agre = b5_scores["Agreeableness"]
    neur = b5_scores["Neuroticism"]
    cons = b5_scores["Conscientiousness"]

    # Raw IPC estimates (based on DeYoung, 2012)
    #raw_agency = 0.60 * extr + 0.25 * agre - 0.20 * neur + 0.10 * cons
    #raw_communion = 0.60 * agre + 0.30 * extr - 0.10 * neur

    #print("a_raw: ", raw_agency, " c_raw: ", raw_communion)

    # Normalize to [0, 1] using derived theoretical min/max
    #norm_agency = (raw_agency + 0.20) / 1.15
    #norm_communion = (raw_communion + 0.10) / 1.00

    #print("a_norm: ", norm_agency, " c_norm: ", norm_communion)

    # Scale to [0, 4]
    #agency_4 = norm_agency * 4
    #communion_4 = norm_communion * 4

    agency_4 = round(extr * 4, 0)
    communion_4 = round(agre * 4, 0)

    print("agency_4: ", agency_4, " communion_4: ", communion_4)

    # return {
    #     "Agency (0–4)": round(agency_4, 2),
    #     "Communion (0–4)": round(communion_4, 2)
    # }
    return [int(agency_4), int(communion_4)]

def rut(text):
    return compute_ipc_from_big5(personality_detection(text))

if __name__ == "__main__":
    text = "sorry sir, is it by any chance possible i could toalk to you?"
    b5_scores = personality_detection(text)
    print(b5_scores)
    ipc_scores = compute_ipc_from_big5(b5_scores)
    print("Final IPC Scores:", ipc_scores)
