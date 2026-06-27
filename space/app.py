import gradio as gr
from transformers import pipeline

MODEL = "Anonym-IA/V2-camembert-ner-pii-french"

ner = pipeline("token-classification", model=MODEL, aggregation_strategy="simple")

LABEL_FR = {
    "PER": "Personne", "PERSON": "Personne",
    "ORG": "Organisation", "ORGANIZATION": "Organisation",
    "LOC": "Lieu", "LOCATION": "Lieu", "GPE": "Lieu",
    "EMAIL": "Email", "PHONE": "Téléphone", "TEL": "Téléphone",
    "DATE": "Date", "ADDRESS": "Adresse", "ADDR": "Adresse",
    "MISC": "Divers",
}

FAKE = {
    "PER":   ["Alice Martin", "Pierre Moreau", "Sophie Bernard", "Luc Dubois", "Emma Leroy"],
    "ORG":   ["Société Dupuis SA", "Cabinet Lemaire", "Groupe Horizon"],
    "LOC":   ["Lyon", "Bordeaux", "Lille", "Nantes", "Strasbourg"],
    "EMAIL": ["contact@exemple.fr", "info@domaine.fr"],
    "PHONE": ["01 23 45 67 89", "06 00 11 22 33"],
    "DATE":  ["le 1er janvier 2000", "le 15 mars 1990"],
    "ADDR":  ["10 avenue des Fleurs, 69000 Lyon"],
    "MISC":  ["[RÉFÉRENCE]"],
}


def normalize_label(raw):
    return LABEL_FR.get(raw.upper(), raw)


def pseudonymize(text, mode):
    if not text.strip():
        return "", "Entrez du texte."

    entities = ner(text)
    if not entities:
        return text, "Aucune entité PII détectée."

    entities = sorted(entities, key=lambda e: e["start"], reverse=True)

    counters = {}
    word_map = {}
    result = text
    details = []

    for ent in sorted(entities, key=lambda e: e["start"]):
        key = ent["word"].strip().lower()
        label = ent["entity_group"]
        label_norm = list(LABEL_FR.keys())[
            next((i for i, k in enumerate(LABEL_FR.keys()) if k == label.upper()), 0)
        ] if label.upper() in LABEL_FR else label

        if key not in word_map:
            counters[label] = counters.get(label, 0) + 1
            n = counters[label]
            if mode == "Balises [ENTITÉ]":
                rep = f"[{normalize_label(label).upper()}_{n}]"
            else:
                pool = FAKE.get(label.upper(), FAKE.get(label, ["[INCONNU]"]))
                rep = pool[(n - 1) % len(pool)]
            word_map[key] = rep

        details.append({
            "type": normalize_label(label),
            "original": ent["word"],
            "remplacement": word_map[key],
            "confiance": f"{round(ent['score'] * 100)}%",
        })

    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        key = ent["word"].strip().lower()
        rep = word_map.get(key, f"[{ent['entity_group']}]")
        result = result[: ent["start"]] + rep + result[ent["end"]:]

    table = "| Type | Original | Remplacement | Confiance |\n|---|---|---|---|\n"
    for d in details:
        table += f"| {d['type']} | *{d['original']}* | **{d['remplacement']}** | {d['confiance']} |\n"

    return result, table


with gr.Blocks(
    title="PortAnonyme — Pseudonymisation PII français",
    theme=gr.themes.Base(
        primary_hue="violet",
        neutral_hue="slate",
    ),
    css="""
        .gradio-container { max-width: 900px !important; }
        #title { text-align: center; margin-bottom: 8px; }
        #subtitle { text-align: center; color: #94a3b8; font-size: 14px; margin-bottom: 24px; }
    """,
) as demo:
    gr.Markdown("# PortAnonyme", elem_id="title")
    gr.Markdown(
        "Pseudonymisation de données personnelles en français · `Anonym-IA/V2-camembert-ner-pii-french`",
        elem_id="subtitle",
    )

    with gr.Row():
        mode = gr.Radio(
            ["Balises [ENTITÉ]", "Pseudonymes fictifs"],
            value="Balises [ENTITÉ]",
            label="Mode",
        )

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Texte original",
                placeholder="Collez ici votre texte en français…\n\nExemple : Bonjour, je m'appelle Jean Dupont et mon email est jean.dupont@example.fr.",
                lines=10,
            )
            btn = gr.Button("Pseudonymiser", variant="primary")

        with gr.Column():
            output_text = gr.Textbox(label="Texte pseudonymisé", lines=10, interactive=False)

    entities_table = gr.Markdown(label="Entités détectées")

    btn.click(
        fn=pseudonymize,
        inputs=[input_text, mode],
        outputs=[output_text, entities_table],
    )

    gr.Examples(
        examples=[
            ["Bonjour, je m'appelle Jean Dupont et mon email est jean.dupont@example.fr. Je réside au 12 rue de la Paix, 75001 Paris. Mon numéro est le 06 12 34 56 78.", "Balises [ENTITÉ]"],
            ["La réunion entre Marie Curie de chez Renault et Pierre Martin d'Airbus aura lieu le 15 janvier à Toulouse.", "Pseudonymes fictifs"],
        ],
        inputs=[input_text, mode],
    )

demo.launch()
