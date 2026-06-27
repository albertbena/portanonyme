import re
import gradio as gr
from transformers import pipeline

MODEL = "Anonym-IA/V2-camembert-ner-pii-french"
ner = pipeline("token-classification", model=MODEL, aggregation_strategy="simple")

# ─── Label → Groupe normalisé ─────────────────────────────────────────────────
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
    if L in {"NUM_SECURITE_SOCIALE", "NSS", "CARTE_IDENTITE", "PASSWORD", "IBAN", "BIC", "IDENTIFIANT", "NUM_CARTE_CREDIT"}:
        return "IDENTIFIANT"
    if L in {"DATE", "DATE_NAISSANCE"}:
        return "DATE"
    return "DIVERS"

GROUP_EMOJI = {
    "PERSONNE": "👤", "ADRESSE": "📍", "ORGANISATION": "🏢",
    "EMAIL": "📧", "TÉLÉPHONE": "📞", "IDENTIFIANT": "🪪",
    "DATE": "📅", "DIVERS": "🔹",
}

# ─── Regex fallbacks (ce que le NER rate souvent) ────────────────────────────
REGEX_FALLBACKS = [
    ("EMAIL",       r"[\w.+%-]+@[\w.-]+\.[a-zA-Z]{2,}"),
    ("TÉLÉPHONE",   r"(?:(?:\+33|0033)\s?[1-9]|0[1-9])(?:[\s.\-]?\d{2}){4}"),
    ("IBAN",        r"\bFR\s?\d{2}(?:\s?\d{4}){5}\s?\d{3}\b"),
    ("IDENTIFIANT", r"\b[A-Z]{6}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b"),  # BIC/SWIFT
    ("IDENTIFIANT", r"\b\d{3}\s\d{3}\s\d{3}\b"),  # RCS / SIREN formaté
    ("IDENTIFIANT", r"\b\d{13,15}\b"),             # SS, CNI
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
    "EMAIL":        ["contact@exemple.fr", "info@domaine.fr", "user@messagerie.fr",
                     "service@entreprise.fr"],
    "TÉLÉPHONE":    ["01 23 45 67 89", "06 00 11 22 33", "04 56 78 90 12", "07 99 88 77 66"],
    "IDENTIFIANT":  ["XX-XXX-XXX", "REF-2024-001", "BIC-XXXXXX", "ID-000000"],
    "DATE":         ["01/01/1990", "15/06/1985", "22/03/2001", "10/11/1978"],
    "DIVERS":       ["[RÉFERENCE]"],
}

# ─── Étape 1 : NER ligne par ligne ────────────────────────────────────────────
def run_ner_by_lines(text):
    entities = []
    pos = 0
    for line in text.split("\n"):
        if line.strip():
            try:
                for e in ner(line):
                    entities.append({
                        "start": pos + e["start"],
                        "end":   pos + e["end"],
                        "label": e["entity_group"],
                        "word":  e["word"],
                        "score": e["score"],
                        "group": group_of(e["entity_group"]),
                    })
            except Exception:
                pass
        pos += len(line) + 1  # +1 pour le \n
    return entities

# ─── Étape 2 : Regex fallbacks ────────────────────────────────────────────────
def apply_regex_fallbacks(text, existing):
    covered = [(e["start"], e["end"]) for e in existing]

    def overlaps(s, e):
        return any(cs <= s < ce or cs < e <= ce or (s <= cs and e >= ce) for cs, ce in covered)

    extra = []
    for label, pattern in REGEX_FALLBACKS:
        for m in re.finditer(pattern, text):
            s, e = m.start(), m.end()
            if not overlaps(s, e):
                extra.append({
                    "start": s, "end": e,
                    "label": label, "word": m.group(),
                    "score": 1.0, "group": group_of(label),
                })
                covered.append((s, e))
    return extra

# ─── Étape 3 : Fusion des fragments adjacents ─────────────────────────────────
def merge_adjacent(entities, text):
    if not entities:
        return entities

    srt = sorted(entities, key=lambda e: e["start"])
    merged = []
    cur = dict(srt[0])

    for nxt in srt[1:]:
        gap = nxt["start"] - cur["end"]
        gap_text = text[cur["end"]:nxt["start"]] if gap > 0 else ""

        same_group = cur["group"] == nxt["group"]

        # Fusionner noms composés (Jean-François, Lefebvre-Moreau) : gap court, tiret ou espace
        is_name_join = (
            same_group and cur["group"] == "PERSONNE"
            and gap <= 3
            and re.match(r'^[\s\-–]*$', gap_text)
        )

        # Fusionner composants d'adresse (numéro + rue + CP + ville) : gap court, séparateurs légers
        is_addr_join = (
            same_group and cur["group"] == "ADRESSE"
            and gap <= 6
            and re.match(r'^[\s,./\-]*$', gap_text)
        )

        # Fusionner prénom + nom séparés par un espace
        is_person_parts = (
            same_group and cur["group"] == "PERSONNE"
            and gap <= 2
            and re.match(r'^\s*$', gap_text)
        )

        if is_name_join or is_addr_join or is_person_parts:
            cur["end"]   = nxt["end"]
            cur["word"]  = text[cur["start"]:nxt["end"]]
            cur["score"] = (cur["score"] + nxt["score"]) / 2
        else:
            merged.append(cur)
            cur = dict(nxt)

    merged.append(cur)
    return merged

# ─── Étape 4 : Suppression des chevauchements ─────────────────────────────────
def remove_overlaps(entities):
    srt = sorted(entities, key=lambda e: e["start"])
    result = []
    last_end = -1
    for e in srt:
        if e["start"] >= last_end:
            result.append(e)
            last_end = e["end"]
        elif e["score"] > result[-1]["score"]:
            result[-1] = e
            last_end = e["end"]
    return result

# ─── Étape 5 : Filtrer les faux positifs évidents ────────────────────────────
STOPWORDS = {
    "le", "la", "les", "de", "du", "des", "un", "une", "au", "aux",
    "en", "et", "ou", "sur", "par", "pour", "dans", "avec", "sans",
    "entre", "sont", "est", "ce", "que", "qui", "se", "ne", "pas",
    "article", "objet", "contrat", "maître", "monsieur", "madame",
    "société", "cabinet", "groupe", "sarl", "sas", "sa", "sasu",
    "avocate", "avocat", "directeur", "général", "née", "né",
}

def filter_entities(entities):
    cleaned = []
    for e in entities:
        word = e["word"].strip()
        # Ignorer les mots trop courts ou stopwords
        if len(word) <= 1:
            continue
        if word.lower() in STOPWORDS:
            continue
        # Ignorer PERSONNE avec score < 0.6 et mot en minuscules
        if e["group"] == "PERSONNE" and e["score"] < 0.6 and word[0].islower():
            continue
        cleaned.append(e)
    return cleaned

# ─── Pipeline complet ─────────────────────────────────────────────────────────
def find_entities(text):
    ner_ents    = run_ner_by_lines(text)
    regex_ents  = apply_regex_fallbacks(text, ner_ents)
    all_ents    = ner_ents + regex_ents
    merged      = merge_adjacent(all_ents, text)
    clean       = remove_overlaps(merged)
    filtered    = filter_entities(clean)
    return sorted(filtered, key=lambda e: e["start"])

# ─── Construction des remplacements ──────────────────────────────────────────
def build_replacements(entities, mode):
    group_counters = {}
    word_to_rep    = {}
    mappings       = []

    for ent in entities:
        key = ent["word"].strip().lower()
        if key in word_to_rep:
            rep = word_to_rep[key]
        else:
            g = ent["group"]
            group_counters[g] = group_counters.get(g, 0) + 1
            n = group_counters[g]

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
        return text, "✅ Aucune entité PII détectée dans ce texte."

    mappings      = build_replacements(entities, mode)
    pseudonymized = apply_replacements(text, mappings)

    # Tableau récapitulatif (dédupliqué)
    seen = set()
    rows = []
    for m in mappings:
        key = (m["word"].strip(), m["replacement"])
        if key not in seen:
            seen.add(key)
            emoji = GROUP_EMOJI.get(m["group"], "•")
            conf  = round(m["score"] * 100)
            rows.append(
                f"| {emoji} **{m['group']}** | {m['word'].strip()} | `{m['replacement']}` | {conf}% |"
            )

    n_unique = len(seen)
    table = (
        f"### {n_unique} entité(s) détectée(s)\n\n"
        "| Type | Original | Remplacement | Confiance |\n"
        "|---|---|---|---|\n"
        + "\n".join(rows)
    )

    return pseudonymized, table

# ─── Interface Gradio ─────────────────────────────────────────────────────────
with gr.Blocks(
    title="PortAnonyme — Pseudonymisation PII français",
    theme=gr.themes.Base(primary_hue="violet", neutral_hue="slate"),
    css="""
        .gradio-container { max-width: 980px !important; }
        #title    { text-align: center; margin-bottom: 4px; }
        #subtitle { text-align: center; color: #94a3b8; font-size: 13px; margin-bottom: 20px; }
    """,
) as demo:
    gr.Markdown("# PortAnonyme", elem_id="title")
    gr.Markdown(
        "Pseudonymisation PII en français · NER `Anonym-IA/V2-camembert-ner-pii-french` + regex",
        elem_id="subtitle",
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
                show_copy_button=True,
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
                "Mon IBAN est FR76 3000 4028 3700 0100 0674 382.",
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

demo.launch()
