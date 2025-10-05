from raw_rules import (
    fallback_en, fallback_de, fallback_fr, 
    fallback_it, fallback_pt, fallback_es, fallback_hi, fallback_th
)
# from lemminflect import getInflection
from mlconjug3 import Conjugator
try:
    from pattern.en import conjugate as en_conj, PRESENT as EN_PRES, PAST as EN_PAST, FUTURE as EN_FUT, SG as EN_SG, PL as EN_PL
    from pattern.de import conjugate as de_conj, PRESENT as DE_PRES, PAST as DE_PAST, FUTURE as DE_FUT, SG as DE_SG, PL as DE_PL
    from pattern.fr import conjugate as fr_conj, PRESENT as FR_PRES, PAST as FR_PAST, FUTURE as FR_FUT, SG as FR_SG, PL as FR_PL
    from pattern.es import conjugate as es_conj, PRESENT as ES_PRES, PAST as ES_PAST, FUTURE as ES_FUT, SG as ES_SG, PL as ES_PL
    from pattern.it import conjugate as it_conj, PRESENT as IT_PRES, PAST as IT_PAST, FUTURE as IT_FUT, SG as IT_SG, PL as IT_PL
except ImportError:
    # If Pattern isn’t installed, set all to None
    en_conj = de_conj = fr_conj = es_conj = it_conj = None
from functools import lru_cache

# Constants at module top
_PT_TENSE_MAP = {
    "present": ("Indicativo", "Indicativo presente"),
    "past":    ("Indicativo", "Indicativo pretérito perfeito simples"),
    "future":  ("Indicativo", "Indicativo Futuro do Presente Simples"),
}

_PT_PRONOUN_MAP = {
    ("1","Sing"): "eu",   ("2","Sing"): "tu",     ("3","Sing"): "ele",
    ("1","Plur"): "nós",  ("2","Plur"): "vós",    ("3","Plur"): "eles",
}

@lru_cache(maxsize=None)
def get_mlconj_form_pt(lemma, tense, person, number):
    conj = Conjugator(language="pt")
    verb = conj.conjugate(lemma)
    target = _PT_TENSE_MAP.get(tense)
    pron   = _PT_PRONOUN_MAP.get((person, number))
    if not target or not pron:
        return None
    # Short‑circuit with next()
    return next(
        (form for mood, tname, p, form in verb.iterate()
         if (mood.strip(), tname.strip(), p.strip()) == (*target, pron)),
        None
    )

@lru_cache(maxsize=None)
def _get_conjugator(lemma):
    return Conjugator(language="pt")

def transform_token(token, tense, lang):
    """Generic dispatcher: try library, then fallback."""
    if token["upostag"] != "VERB":
        return token["form"]

    # lemma = token.get("lemma", token["form"])
    lemma = token.get("lemma", token.get("form", "")).lower()
    feats = token.get("feats", {})
    if feats:
        person = int(feats.get("Person", "3"))
        number = feats.get("Number", "Sing")
        mood   = feats.get("Mood", "Ind")
        # use any module’s SG/PL
        # number = ES_SG if feats.get("Number","Sing")=="Sing" else ES_PL

    try:
        if lang == "en" and en_conj:
            number_map = {
                ("Sing"): EN_SG,  ("Plur"): EN_PL
            }
            number = number_map.get((number), EN_SG)
            verb = en_conj(lemma, {"present":EN_PRES, "past":EN_PAST, "future":EN_FUT}[tense], person, number)
            # print(f"------ {tense} ------")
            # print(f"verb = {lemma}")
            # print(f"infl = {verb}")
            if verb:
                return verb
            else:
                raise ValueError
            
        if lang == "de" and de_conj:
            number_map = {
                ("Sing"): DE_SG,  ("Plur"): DE_PL
            }
            number = number_map.get((number), DE_SG)

            verb = de_conj(lemma, {"present":DE_PRES, "past":DE_PAST, "future":DE_FUT}[tense], person, number)
            # print(f"------ {tense} ------")
            # print(f"verb = {lemma}")
            # print(f"infl = {verb}")
            if verb:
                return verb
            else:
                raise ValueError
            
        if lang == "fr" and fr_conj:
            number_map = {
                ("Sing"): FR_SG,  ("Plur"): FR_PL
            }
            number = number_map.get((number), FR_SG)

            verb = fr_conj(lemma, {"present":FR_PRES, "past":FR_PAST, "future":FR_FUT}[tense], person, number)
            if verb:
                return verb
            else:
                raise ValueError
            
        if lang == "es" and es_conj:
            number_map = {
                ("Sing"): ES_SG,  ("Plur"): ES_PL
            }
            number = number_map.get((number), ES_SG)

            verb = es_conj(lemma, {"present":ES_PRES, "past":ES_PAST, "future":ES_FUT}[tense], person, number)
            if verb:
                return verb
            else:
                raise ValueError
            
        if lang == "it" and it_conj:
            number_map = {
                ("Sing"): IT_SG,  ("Plur"): IT_PL
            }
            number = number_map.get((number), IT_SG)

            verb = it_conj(lemma, {"present":IT_PRES, "past":IT_PAST, "future":IT_FUT}[tense], person, number)
            if verb:
                return verb
            else:
                raise ValueError
        
        elif lang == "pt":
            # initialize the conjugator
            conj = _get_conjugator(lemma)

            lib_form = get_mlconj_form_pt(conj, lemma, tense, person, number)
            # print(f"------ {tense} ------")
            # print(f"verb = {lemma}")
            # print(f"infl = {lib_form}")
            if lib_form:
                return lib_form
            else:
                raise ValueError
                
        elif lang == "hi":
            # no Hindi conjugator: fallback only
            raise NotImplementedError
        
        elif lang == "th":
            # Thai verbs uninflected: particles
            raise NotImplementedError
        
        else:
            raise NotImplementedError

    except Exception:
        return {
            "en": fallback_en,
            "de": fallback_de,
            "fr": fallback_fr,
            "es": fallback_es,
            "it": fallback_it,
            "pt": fallback_pt,
            "hi": fallback_hi,
            "th": fallback_th
        }[lang](token, tense)

# Configuration mapping unchanged:
LANGUAGE_CONFIG = {
    'en': {'transform': lambda t, ts, _: transform_token(t, ts, 'en')},
    'de': {'transform': lambda t, ts, _: transform_token(t, ts, 'de')},
    'fr': {'transform': lambda t, ts, _: transform_token(t, ts, 'fr')},
    'it': {'transform': lambda t, ts, _: transform_token(t, ts, 'it')},
    'pt': {'transform': lambda t, ts, _: transform_token(t, ts, 'pt')},
    'es': {'transform': lambda t, ts, _: transform_token(t, ts, 'es')},
    'hi': {'transform': lambda t, ts, _: transform_token(t, ts, 'hi')},
    'th': {'transform': lambda t, ts, _: transform_token(t, ts, 'th')},
}
    