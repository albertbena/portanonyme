import re
import gradio as gr
from transformers import pipeline

MODEL = "Anonym-IA/V2-camembert-ner-pii-french"
ner = pipeline("token-classification", model=MODEL, aggregation_strategy="simple")

# ─── Normalisation des labels ─────────────────────────────────────────────────
def group_of(label):
    L = label.upper()
    if L in {"PRENOM_PERSONNE", "NOM_PERSONNE", "PER", "PERSON", "NOM", "PRENOM"}:
        return "PERSONNE"
    if L in {"NUMERO_VOIE", "NOM_VOIE", "CODE_POSTAL", "VILLE", "LOC", "LOCATION", "GPE", "ADRESSE", "ADDRESS"}:
        return "ADRESSE"
    if L in {"NOM_SOCIETE", "ORG", "ORGANIZATION", "ORGANISATION"}:
        return "ORGANISATION"
    if L in {"EMAIL", "COURRIEL", "MAIL"}:
        return "EMAIL"
    if L in {"TELEPHONE", "TEL", "PHONE", "MOBILE"}:
        return "TÉLÉPHONE"
    if L in {"NUM_SECURITE_SOCIALE", "NSS", "CARTE_IDENTITE", "PASSWORD",
             "IBAN", "BIC", "IDENTIFIANT", "NUM_CARTE_CREDIT"}:
        return "IDENTIFIANT"
    if L in {"DATE", "DATE_NAISSANCE"}:
        return "DATE"
    return "DIVERS"

GROUP_EMOJI = {
    "PERSONNE": "👤", "ADRESSE": "📍", "ORGANISATION": "🏢",
    "EMAIL": "📧", "TÉLÉPHONE": "📞", "IDENTIFIANT": "🪪",
    "DATE": "📅", "DIVERS": "🔹",
}

# ─── Regex haute précision (toujours prioritaires sur le NER) ─────────────────
# BIC : 4 lettres banque + code pays ISO valide + 2 chars + 3 optionnel
_CC = (
    "AD|AE|AT|AU|BE|BH|BR|BY|CA|CH|CN|CY|CZ|DE|DK|DZ|EG|ES|FI|FR|GB|GR|"
    "HK|HR|HU|ID|IE|IL|IN|IS|IT|JP|KW|KZ|LB|LI|LT|LU|LV|MA|MC|ME|MT|MX|"
    "NL|NO|NZ|OM|PL|PT|QA|RO|RS|RU|SA|SE|SG|SI|SK|SM|TN|TR|UA|US|ZA"
)
_BIC = rf"[A-Z]{{4}}(?:{_CC})[A-Z0-9]{{2}}(?:[A-Z0-9]{{3}})?"

REGEX_RULES = [
    # Emails complets (avant tout autre regex)
    ("EMAIL",       r"[\w.+%-]+@[\w.-]+\.[a-zA-Z]{2,}"),
    # Téléphones français
    ("TÉLÉPHONE",   r"(?:(?:\+33|0033)\s?[1-9]|0[1-9])(?:[\s.\-]?\d{2}){4}"),
    # IBAN français
    ("IDENTIFIANT", r"\bFR\s?\d{2}(?:\s?\d{4}){5}\s?\d{3}\b"),
    # BIC/SWIFT avec validation code pays
    ("IDENTIFIANT", _BIC),
    # RCS/SIREN (3 groupes de 3 chiffres séparés par espaces)
    ("IDENTIFIANT", r"\b\d{3}\s\d{3}\s\d{3}\b"),
    # CNI / numéro SS (12 à 15 chiffres collés)
    ("IDENTIFIANT", r"\b\d{12,15}\b"),
]

# ─── Pseudonymes fictifs ──────────────────────────────────────────────────────
FAKE = {
    "PERSONNE":     ["Alice Martin", "Pierre Moreau", "Sophie Bernard", "Luc Dubois",
                     "Emma Leroy", "Thomas Petit", "Camille Roux", "Julien Faure"],
    "ADRESSE":      ["10 avenue des Fleurs, 69001 Lyon", "5 rue des Lilas, 33000 Bordeaux",
                     "22 bd Gambetta, 59000 Lille", "8 place du Capitole, 31000 Toulouse",
                     "3 impasse des Roses, 44000 Nantes"],
    "ORGANISATION": ["Société Lumière SAS", "Cabinet Horizon SARL",
                     "Entreprise Leblanc & Fils", "Groupe Éclipse SA", "SARL Avenir"],
    "EMAIL":        ["contact@exemple.fr", "info@domaine.fr",
                     "user@messagerie.fr", "service@entreprise.fr"],
    "TÉLÉPHONE":    ["01 23 45 67 89", "06 00 11 22 33", "04 56 78 90 12", "07 99 88 77 66"],
    "IDENTIFIANT":  ["XX-XXX-XXX", "REF-2024-001", "BIC-XXXXXX", "ID-000000"],
    "DATE":         ["01/01/1990", "15/06/1985", "22/03/2001", "10/11/1978"],
    "DIVERS":       ["[RÉFÉRENCE]"],
}

# Mots à ne jamais pseudonymiser
STOPWORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "au", "aux", "en", "et",
    "ou", "sur", "par", "pour", "dans", "avec", "sans", "entre", "sont", "est",
    "article", "objet", "contrat", "service", "services", "prestation", "prestations",
    "maître", "monsieur", "madame", "avocate", "avocat",
    "directeur", "général", "née", "né", "drh", "cgt", "cfe", "cfdt",
}

# Préfixes de voies (pour détecter les noms de rues)
STREET_PREFIX = re.compile(
    r"\b(?:rue|avenue|av\.|bd\.?|boulevard|allée|impasse|place|chemin|route|"
    r"passage|square|villa|résidence|hameau|cité|voie|ruelle)\s+",
    re.IGNORECASE,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def overlaps(s1, e1, s2, e2):
    return s1 < e2 and s2 < e1


def make_ent(start, end, label, word, score):
    return {"start": start, "end": end, "label": label,
            "word": word, "score": score, "group": group_of(label)}


# ─── Étape 1 : Regex (haute précision, prioritaire) ──────────────────────────
def run_regex(text):
    entities = []
    covered  = []
    for label, pattern in REGEX_RULES:
        for m in re.finditer(pattern, text):
            s, e = m.start(), m.end()
            if not any(overlaps(s, e, cs, ce) for cs, ce in covered):
                entities.append(make_ent(s, e, label, m.group(), 1.0))
                covered.append((s, e))
    return entities


# ─── Étape 2 : NER ligne par ligne ───────────────────────────────────────────
def run_ner(text):
    entities = []
    pos = 0
    for line in text.split("\n"):
        if line.strip():
            try:
                for e in ner(line):
                    entities.append(make_ent(
                        pos + e["start"], pos + e["end"],
                        e["entity_group"], e["word"], e["score"],
                    ))
            except Exception:
                pass
        pos += len(line) + 1
    return entities


# ─── Étape 3 : Re-labellisation contextuelle ─────────────────────────────────
def relabel_context(entities, text):
    result = []
    for e in entities:
        ent = dict(e)
        if ent["group"] == "PERSONNE":
            # Nom propre précédé d'un préfixe de voie → c'est un nom de rue
            before = text[max(0, ent["start"] - 50): ent["start"]]
            if STREET_PREFIX.search(before):
                ent["group"] = "ADRESSE"
                ent["label"] = "ADRESSE"
        result.append(ent)
    return result


# ─── Étape 4 : Fusion des fragments adjacents ────────────────────────────────
def merge_adjacent(entities, text):
    if not entities:
        return entities

    srt    = sorted(entities, key=lambda e: e["start"])
    merged = []
    cur    = dict(srt[0])

    for nxt in srt[1:]:
        gap      = nxt["start"] - cur["end"]
        gap_text = text[cur["end"]: nxt["start"]] if gap > 0 else ""
        same_g   = cur["group"] == nxt["group"]

        # Prénom + nom (Jean-François Dupont, Lefebvre-Moreau…)
        is_name = (
            same_g and cur["group"] == "PERSONNE"
            and gap <= 3
            and re.match(r'^[\s\-–]*$', gap_text)
        )
        # Composants d'adresse adjacents
        is_addr = (
            same_g and cur["group"] == "ADRESSE"
            and gap <= 8
            and re.match(r'^[\s,./\-\d]*$', gap_text)
        )

        if is_name or is_addr:
            cur["end"]   = nxt["end"]
            cur["word"]  = text[cur["start"]: nxt["end"]]
            cur["score"] = (cur["score"] + nxt["score"]) / 2
        else:
            merged.append(cur)
            cur = dict(nxt)

    merged.append(cur)
    return merged


# ─── Étape 5 : Dédoublonnage / chevauchements ────────────────────────────────
def remove_overlaps(entities):
    srt    = sorted(entities, key=lambda e: e["start"])
    result = []
    last   = -1
    for e in srt:
        if e["start"] >= last:
            result.append(e)
            last = e["end"]
        elif e["score"] > result[-1]["score"]:
            result[-1] = e
            last = e["end"]
    return result


# ─── Étape 6 : Filtrage des faux positifs ────────────────────────────────────
def filter_fp(entities):
    out = []
    for e in entities:
        w = e["word"].strip()
        if len(w) <= 1:
            continue
        if w.lower() in STOPWORDS:
            continue
        if e["group"] == "PERSONNE" and e["score"] < 0.65 and w[0].islower():
            continue
        out.append(e)
    return out


# ─── Pipeline complet ─────────────────────────────────────────────────────────
def find_entities(text):
    # 1. Regex en premier (emails, téléphones, IBAN, BIC…)
    regex_ents  = run_regex(text)
    regex_spans = [(e["start"], e["end"]) for e in regex_ents]

    # 2. NER ligne par ligne
    ner_ents = run_ner(text)

    # 3. Rejeter les entités NER qui chevauchent les regex (le regex est plus fiable)
    ner_clean = [
        e for e in ner_ents
        if not any(overlaps(e["start"], e["end"], rs, re_) for rs, re_ in regex_spans)
    ]

    # 4. Re-labellisation contextuelle (noms de rues)
    ner_relabeled = relabel_context(ner_clean, text)

    # 5. Fusion des fragments adjacents
    ner_merged = merge_adjacent(ner_relabeled, text)

    # 6. Combiner regex + NER
    all_ents = regex_ents + ner_merged

    # 7. Supprimer les chevauchements restants
    clean = remove_overlaps(all_ents)

    # 8. Filtrer les faux positifs
    return sorted(filter_fp(clean), key=lambda e: e["start"])


# ─── Construction des remplacements ──────────────────────────────────────────
def build_replacements(entities, mode):
    counters   = {}
    word_to_rep = {}
    mappings   = []

    for ent in entities:
        key = ent["word"].strip().lower()
        if key in word_to_rep:
            rep = word_to_rep[key]
        else:
            g            = ent["group"]
            counters[g]  = counters.get(g, 0) + 1
            n            = counters[g]
            if mode == "Balises [ENTITÉ]":
                rep = f"[{g}_{n}]"
            else:
                pool = FAKE.get(g, FAKE["DIVERS"])
                rep  = pool[(n - 1) % len(pool)]
            word_to_rep[key] = rep

        mappings.append({**ent, "replacement": rep})

    return mappings


def apply_replacements(text, mappings):
    result = text
    for m in sorted(mappings, key=lambda e: e["start"], reverse=True):
        result = result[:m["start"]] + m["replacement"] + result[m["end"]:]
    return result


# ─── Fonction principale ──────────────────────────────────────────────────────
def pseudonymize(text, mode):
    if not text.strip():
        return "", "Entrez du texte à pseudonymiser."

    entities = find_entities(text)

    if not entities:
        return text, "✅ Aucune entité PII détectée."

    mappings      = build_replacements(entities, mode)
    pseudonymized = apply_replacements(text, mappings)

    seen = set()
    rows = []
    for m in mappings:
        key = (m["word"].strip(), m["replacement"])
        if key not in seen:
            seen.add(key)
            emoji = GROUP_EMOJI.get(m["group"], "•")
            rows.append(
                f"| {emoji} **{m['group']}** "
                f"| {m['word'].strip()} "
                f"| `{m['replacement']}` "
                f"| {round(m['score'] * 100)}% |"
            )

    table = (
        f"### {len(seen)} entité(s) détectée(s)\n\n"
        "| Type | Original | Remplacement | Confiance |\n"
        "|---|---|---|---|\n"
        + "\n".join(rows)
    )

    return pseudonymized, table


# ─── Interface Gradio ─────────────────────────────────────────────────────────
with gr.Blocks(title="PortAnonyme — Pseudonymisation PII français") as demo:
    gr.Markdown("# PortAnonyme")
    gr.Markdown(
        "Pseudonymisation PII en français · NER CamemBERT + regex",
        elem_classes=["subtitle"],
    )

    mode = gr.Radio(
        ["Balises [ENTITÉ]", "Pseudonymes fictifs"],
        value="Balises [ENTITÉ]",
        label="Mode de remplacement",
    )

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Texte original",
                placeholder="Collez votre texte français ici…",
                lines=16,
            )
            btn = gr.Button("🔒 Pseudonymiser", variant="primary", size="lg")

        with gr.Column():
            output_text = gr.Textbox(
                label="Texte pseudonymisé",
                lines=16,
                interactive=False,
            )

    entities_md = gr.Markdown()

    btn.click(
        fn=pseudonymize,
        inputs=[input_text, mode],
        outputs=[output_text, entities_md],
    )

    gr.Examples(
        examples=[
            [
                "Bonjour, je m'appelle Jean-François Dupont et mon email est jf.dupont@example.fr. "
                "Je réside au 12 rue de la Paix, 75001 Paris. Mon numéro est le 06 12 34 56 78. "
                "Mon IBAN est FR76 3000 4028 3700 0100 0674 382, BIC BNPAFRPPXXX.",
                "Balises [ENTITÉ]",
            ],
            [
                "La réunion entre Marie Curie de chez Renault SA et Pierre-Antoine Martin "
                "d'Airbus Group aura lieu le 15 janvier à Toulouse.",
                "Pseudonymes fictifs",
            ],
        ],
        inputs=[input_text, mode],
    )

demo.launch(
    theme=gr.themes.Base(primary_hue="violet", neutral_hue="slate"),
)
