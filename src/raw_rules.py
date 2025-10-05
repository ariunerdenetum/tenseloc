# --- English transformation functions ---
def fallback_en(token, tense, sentence=None):
    """
    Transforms an English VERB token to the target tense.
    
    - PRESENT: return the base form (lemma).
    - PAST: if token is marked as Past via feats, use the token form; otherwise, naively append "ed".
    - FUTURE: always return "will " + lemma, forcing a simple future form.
    """
    if token["upostag"] != "VERB":
        return token["form"]
    
    # Get lemma; fallback to form if missing.
    lemma = token.get("lemma", token["form"]).lower()
    feats = token.get("feats", {})

    if tense == "present":
        return lemma
    elif tense == "past":
        # if feats.get("Tense") == "Past":
        #     return token["form"]
        # Naively generate past form for regular verbs.
        if lemma.endswith("e"):
            return lemma + "d"
        else:
            return lemma + "ed"
    elif tense == "future":
        # Always generate a simple future form.
        return "will " + lemma
    else:
        return token["form"]

# --- German transformation functions ---
def fallback_de(token, tense, sentence=None):
    """
    Improved German fallback:
    - Present: stem + personal ending
    - Past: passthrough strong (feats), else Präteritum weak endings
    - Future: conjugated werden + lemma
    """
    if token.get("upostag") != "VERB":
        return token.get("form", "")

    lemma = token.get("lemma", token.get("form", ""))
    feats = token.get("feats") or {}
    person = feats.get("Person", "3")   # '1','2','3'
    number = feats.get("Number", "Sing")  # 'Sing','Plur'
    stem = lemma[:-2] if lemma.endswith("en") else lemma

    # Present tense endings
    if tense == "present":
        endings = {
            ("1","Sing"): "e", ("2","Sing"): "st", ("3","Sing"): "t",
            ("1","Plur"): "en", ("2","Plur"): "t",  ("3","Plur"): "en"
        }
        return stem + endings.get((person, number), "en")

    # Past tense: allow irregulars, else weak endings
    if tense == "past":
        if feats.get("Tense") == "Past":
            return token.get("form", lemma)
        endings = {
            ("1","Sing"): "te",   ("2","Sing"): "test", ("3","Sing"): "te",
            ("1","Plur"): "ten",  ("2","Plur"): "tet",  ("3","Plur"): "ten"
        }
        return stem + endings.get((person, number), "ten")

    # Future tense: conjugated 'werden' + infinitive
    if tense == "future":
        werden_forms = {
            ("1","Sing"): "werde", ("2","Sing"): "wirst", ("3","Sing"): "wird",
            ("1","Plur"): "werden",("2","Plur"): "werdet",("3","Plur"): "werden"
        }
        aux = werden_forms.get((person, number), "werden")
        return f"{aux} {lemma}"

    return token.get("form", "")
    
# --- French transformation function ---
def fallback_fr(token, tense, sentence=None):
    """
    Rule-based French inflector for regular verbs:
      - Present: stem + present endings
      - Past: auxiliary 'avoir' (present) + past participle
      - Future: infinitive or stem + future endings
    Uses UD feats Person & Number for agreement.
    """
    if token.get("upostag") != "VERB":
        return token.get("form", "")

    lemma = token.get("lemma", token.get("form", "")).lower()
    feats = token.get("feats") or {}
    person = feats.get("Person", "3")    # '1','2','3'
    number = feats.get("Number", "Sing") # 'Sing','Plur'

    # Identify verb class and stems
    if lemma.endswith("er"):
        stem = lemma[:-2]
        pp_end = "é"
        future_base = lemma
    elif lemma.endswith("ir"):
        stem = lemma[:-2]
        pp_end = "i"
        future_base = lemma
    elif lemma.endswith("re"):
        stem = lemma[:-1]
        pp_end = "u"
        future_base = stem  # drop the 'e'
    else:
        # not a regular verb: leave unchanged
        return token.get("form", "")

    # Present tense endings
    present_endings = {
        ("1","Sing"): "e",  ("2","Sing"): "es", ("3","Sing"): "e",
        ("1","Plur"): "ons",("2","Plur"): "ez", ("3","Plur"): "ent"
    }
    if tense == "present":
        ending = present_endings.get((person, number), "e")
        return stem + ending

    # Passé composé: conjugate 'avoir' + past participle
    if tense == "past":
        # avoir present forms
        aux = {
            ("1","Sing"): "j'ai", ("2","Sing"): "tu as",  ("3","Sing"): "il a",
            ("1","Plur"): "nous avons", ("2","Plur"): "vous avez", ("3","Plur"): "ils ont"
        }.get((person, number), "il a")
        return f"{aux} {stem}{pp_end}"

    # Futur simple: attach endings to infinitive or stem
    future_endings = {
        ("1","Sing"): "erai",  ("2","Sing"): "eras",  ("3","Sing"): "era",
        ("1","Plur"): "erons", ("2","Plur"): "erez",  ("3","Plur"): "eront"
    }
    if tense == "future":
        ending = future_endings.get((person, number), "era")
        return future_base + ending

    # Fallback
    return token.get("form", "")

# --- Italian transformation function ---
def fallback_it(token, tense, sentence=None):
    """
    Rule-based Italian fallback for regular verbs:
      - Present: stem + person-specific endings
      - Past: compound past (avere + past participle)
      - Future: infinitive stem + future endings
    Handles -are, -ere, -ire verbs.
    """
    if token.get("upostag") != "VERB":
        return token.get("form", "")

    lemma = token.get("lemma", token.get("form", "")).lower()
    feats = token.get("feats") or {}
    person = feats.get("Person", "3")
    number = feats.get("Number", "Sing")

    # Determine conjugation group
    if lemma.endswith("are"):
        group = "are"
        stem = lemma[:-3]
        pp = stem + "ato"
        future_stem = stem + "er"
    elif lemma.endswith("ere"):
        group = "ere"
        stem = lemma[:-3]
        pp = stem + "uto"
        future_stem = stem + "er"
    elif lemma.endswith("ire"):
        group = "ire"
        stem = lemma[:-3]
        pp = stem + "ito"
        future_stem = stem + "ir"
    else:
        return token.get("form", "")

    # Present tense endings
    present_endings = {
        "are": {
            ("1", "Sing"): "o",  ("2", "Sing"): "i",  ("3", "Sing"): "a",
            ("1", "Plur"): "iamo", ("2", "Plur"): "ate", ("3", "Plur"): "ano"
        },
        "ere": {
            ("1", "Sing"): "o",  ("2", "Sing"): "i",  ("3", "Sing"): "e",
            ("1", "Plur"): "iamo", ("2", "Plur"): "ete", ("3", "Plur"): "ono"
        },
        "ire": {
            ("1", "Sing"): "o",  ("2", "Sing"): "i",  ("3", "Sing"): "e",
            ("1", "Plur"): "iamo", ("2", "Plur"): "ite", ("3", "Plur"): "ono"
        }
    }

    # Future tense endings (same across groups except stem differs)
    future_endings = {
        ("1", "Sing"): "ò",   ("2", "Sing"): "ai",  ("3", "Sing"): "à",
        ("1", "Plur"): "emo", ("2", "Plur"): "ete", ("3", "Plur"): "anno"
    }

    # Avere conjugation (present)
    avere_aux = {
        ("1", "Sing"): "ho",    ("2", "Sing"): "hai",   ("3", "Sing"): "ha",
        ("1", "Plur"): "abbiamo", ("2", "Plur"): "avete", ("3", "Plur"): "hanno"
    }

    if tense == "present":
        ending = present_endings[group].get((person, number), "a")
        return stem + ending

    elif tense == "past":
        aux = avere_aux.get((person, number), "ha")
        return f"{aux} {pp}"

    elif tense == "future":
        ending = future_endings.get((person, number), "à")
        return future_stem + ending

    return token.get("form", "")

# --- Portuguese transformation function ---
def fallback_pt(token, tense, sentence=None):
    """
    Fallback inflection for Portuguese regular verbs:
      - Handles -ar, -er, -ir endings.
      - Uses Person and Number features.
      - Present, past (preterite), and future simple forms.
    """
    if token["upostag"] != "VERB":
        return token.get("form", "")

    lemma = token.get("lemma", token.get("form", "")).lower()
    feats = token.get("feats") or {}
    person = feats.get("Person", "3")
    number = feats.get("Number", "Sing")

    if lemma.endswith("ar"):
        group = "ar"
        stem = lemma[:-2]
    elif lemma.endswith("er"):
        group = "er"
        stem = lemma[:-2]
    elif lemma.endswith("ir"):
        group = "ir"
        stem = lemma[:-2]
    else:
        return token.get("form", "")

    # Present tense endings
    present_endings = {
        "ar": {
            ("1", "Sing"): "o",   ("2", "Sing"): "as",  ("3", "Sing"): "a",
            ("1", "Plur"): "amos", ("2", "Plur"): "ais", ("3", "Plur"): "am"
        },
        "er": {
            ("1", "Sing"): "o",   ("2", "Sing"): "es",  ("3", "Sing"): "e",
            ("1", "Plur"): "emos", ("2", "Plur"): "eis", ("3", "Plur"): "em"
        },
        "ir": {
            ("1", "Sing"): "o",   ("2", "Sing"): "es",  ("3", "Sing"): "e",
            ("1", "Plur"): "imos", ("2", "Plur"): "is",  ("3", "Plur"): "em"
        }
    }

    # Preterite (past simple) endings
    past_endings = {
        "ar": {
            ("1", "Sing"): "ei",  ("2", "Sing"): "aste",  ("3", "Sing"): "ou",
            ("1", "Plur"): "amos", ("2", "Plur"): "astes", ("3", "Plur"): "aram"
        },
        "er": {
            ("1", "Sing"): "i",   ("2", "Sing"): "este",  ("3", "Sing"): "eu",
            ("1", "Plur"): "emos", ("2", "Plur"): "estes", ("3", "Plur"): "eram"
        },
        "ir": {
            ("1", "Sing"): "i",   ("2", "Sing"): "iste",  ("3", "Sing"): "iu",
            ("1", "Plur"): "imos", ("2", "Plur"): "istes", ("3", "Plur"): "iram"
        }
    }

    # Future tense endings (same for all groups)
    future_endings = {
        ("1", "Sing"): "ei",   ("2", "Sing"): "ás",  ("3", "Sing"): "á",
        ("1", "Plur"): "emos", ("2", "Plur"): "eis", ("3", "Plur"): "ão"
    }

    if tense == "present":
        ending = present_endings[group].get((person, number), "a")
        return stem + ending

    elif tense == "past":
        ending = past_endings[group].get((person, number), "ou")
        return stem + ending

    elif tense == "future":
        ending = future_endings.get((person, number), "á")
        return lemma + ending

    return token.get("form", "")

# --- Spanish transformation function ---
def fallback_es(token, tense, sentence=None):
    """
    Improved fallback inflection for regular Spanish verbs.
    Handles:
      - Present, past (preterite), future tenses.
      - -ar, -er, -ir conjugations.
      - Person and Number agreement from token feats.
    """
    if token["upostag"] != "VERB":
        return token.get("form", "")

    lemma = token.get("lemma", token.get("form", "")).lower()
    feats = token.get("feats") or {}
    person = feats.get("Person", "3")
    number = feats.get("Number", "Sing")

    if lemma.endswith("ar"):
        group = "ar"
        stem = lemma[:-2]
    elif lemma.endswith("er"):
        group = "er"
        stem = lemma[:-2]
    elif lemma.endswith("ir"):
        group = "ir"
        stem = lemma[:-2]
    else:
        return token.get("form", "")

    # Present tense endings
    present_endings = {
        "ar": {
            ("1", "Sing"): "o",   ("2", "Sing"): "as",  ("3", "Sing"): "a",
            ("1", "Plur"): "amos", ("2", "Plur"): "áis", ("3", "Plur"): "an"
        },
        "er": {
            ("1", "Sing"): "o",   ("2", "Sing"): "es",  ("3", "Sing"): "e",
            ("1", "Plur"): "emos", ("2", "Plur"): "éis", ("3", "Plur"): "en"
        },
        "ir": {
            ("1", "Sing"): "o",   ("2", "Sing"): "es",  ("3", "Sing"): "e",
            ("1", "Plur"): "imos", ("2", "Plur"): "ís",  ("3", "Plur"): "en"
        }
    }

    # Preterite (past) tense endings
    past_endings = {
        "ar": {
            ("1", "Sing"): "é",   ("2", "Sing"): "aste",  ("3", "Sing"): "ó",
            ("1", "Plur"): "amos", ("2", "Plur"): "asteis", ("3", "Plur"): "aron"
        },
        "er": {
            ("1", "Sing"): "í",   ("2", "Sing"): "iste",  ("3", "Sing"): "ió",
            ("1", "Plur"): "imos", ("2", "Plur"): "isteis", ("3", "Plur"): "ieron"
        },
        "ir": {
            ("1", "Sing"): "í",   ("2", "Sing"): "iste",  ("3", "Sing"): "ió",
            ("1", "Plur"): "imos", ("2", "Plur"): "isteis", ("3", "Plur"): "ieron"
        }
    }

    # Future tense endings (same for all conjugations)
    future_endings = {
        ("1", "Sing"): "é",   ("2", "Sing"): "ás",  ("3", "Sing"): "á",
        ("1", "Plur"): "emos", ("2", "Plur"): "éis", ("3", "Plur"): "án"
    }

    if tense == "present":
        ending = present_endings[group].get((person, number), "a")
        return stem + ending

    elif tense == "past":
        ending = past_endings[group].get((person, number), "ó")
        return stem + ending

    elif tense == "future":
        ending = future_endings.get((person, number), "á")
        return lemma + ending

    return token.get("form", "")
    
# --- Hindi transformation function ---
def fallback_hi(token, tense, sentence=None):
    """
    Morphology‑aware Hindi inflector: handles Person, Number & Gender.
    Assumes infinitive ends with 'ना'; falls back to form() otherwise.
    """
    if token.get("upostag") != "VERB":
        return token.get("form", "")

    lemma = token.get("lemma", token.get("form", ""))
    feats = token.get("feats") or {}

    # If no infinitive (-ना), bail out
    if not lemma.endswith("ना"):
        return token.get("form", "")

    # Strip 'ना' to get the stem
    stem = lemma[:-2]

    # UD FEATS
    person = feats.get("Person", "3")    # '1','2','3'
    number = feats.get("Number", "Sing") # 'Sing','Plur'
    gender = feats.get("Gender", "Masc") # 'Masc','Fem'

    # Present Habitual (participial + auxiliary)
    if tense == "present":
        part = ("ते" if number=="Plur" and gender=="Masc"
                else "तीं" if number=="Plur"
                else "ता" if gender=="Masc"
                else "ती")
        aux_map = {
            ("1","Sing"): "हूँ", ("2","Sing"): "हो",  ("3","Sing"): "है",
            ("1","Plur"): "हैं", ("2","Plur"): "हो",  ("3","Plur"): "हैं"
        }
        aux = aux_map.get((person, number), "है")
        return f"{stem}{part} {aux}"

    # Past Simple (participial + past auxiliary)
    if tense == "past":
        part = ("ए" if number=="Plur" and gender=="Masc"
                else "ईं" if number=="Plur"
                else "या" if gender=="Masc"
                else "ई")
        aux_map = {
            ("1","Sing"): "था",  ("2","Sing"): "था",  ("3","Sing"): "था",
            ("1","Plur"): "थे", ("2","Plur"): "थे", ("3","Plur"): "थे"
        }
        aux = aux_map.get((person, number), "था")
        return f"{stem}{part} {aux}"

    if tense == "future":
        if person=="1" and number=="Sing":
            ending = "करूंगा" if gender=="Masc" else "करूंगी"
        elif person=="1" and number=="Plur":
            ending = "करेंगे" if gender=="Masc" else "करेंगे"
        elif person=="2":
            ending = "करोगे" if gender=="Masc" else "करोगी"
        elif person=="3" and number=="Sing":
            ending = "करेगा" if gender=="Masc" else "करेगी"
        elif person=="3" and number=="Plur":
            ending = "करेंगे" if gender=="Masc" else "करेंगी"
        else:
            ending = "करूंगा" if gender=="Masc" else "करूंगी"
        return f"{stem}{ending}"

    # Fallback to the raw form if tense not recognized
    return token.get("form", "")

# --- Thai transformation function ---
def fallback_th(token, tense, sentence=None, aspect=None):
    """
    Thai TAM fallback: inserts appropriate particles/auxiliaries.
    tension: 'present', 'past', 'future'
    aspect: None|'progressive'|'experiential'|'imminent'
    """
    if token.get("upostag") != "VERB":
        return token.get("form", "")

    # Determine lemma (or form) of the verb
    lemma = token.get("form")
    # lemma = token.get("lemma", token.get("form", ""))
    # print(f"lemma = {lemma}")

    # Present / Imperfective
    if tense == "present":
        return lemma
        # if aspect == "progressive":
        #     # ongoing: กำลัง V อยู่
        #     return f"กำลัง{lemma}อยู่"
        # elif aspect == "continuative":
        #     # simple continuous
        #     return f"{lemma}อยู่"
        # else:
        #     # simple habitual or undefined aspect
        #     return lemma

    # Past / Perfective
    if tense == "past":
        return f"{lemma}แล้ว"
        # if aspect == "experiential":
        #     # has-experienced: ได้ V (optionally + แล้ว)
        #     return f"ได้{lemma}"
        # else:
        #     # completed: V แล้ว
        #     return f"{lemma}แล้ว"

    # Future / Prospective
    if tense == "future":
        return f"จะ{lemma}"
        # if aspect == "imminent":
        #     # about to: กำลังจะ V
        #     return f"กำลังจะ{lemma}"
        # else:
        #     # general future: จะ V
        #     return f"จะ{lemma}"

    # Fallback: just return lemma
    return lemma

    
# LANGUAGE_CONFIG = {
#     'en': {
#         'transform': transform_token_english
#     },
#     'de': {
#         'transform': transform_token_german
#     },
#     'fr': {
#         'transform': transform_token_french
#     },
#     'it': {
#         'transform': transform_token_italian
#     },
#     'pt': {
#         'transform': transform_token_portuguese
#     },
#     'hi': {
#         'transform': transform_token_hindi
#     },
#     'es': {
#         'transform': transform_token_spanish
#     },
#     'th': {
#         'transform': transform_token_thai
#     },
# }
