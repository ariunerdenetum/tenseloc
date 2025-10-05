import os
import csv
import argparse
from conllu import parse_incr

# Configuration for different languages.
LANGUAGE_CONFIG = {
    'en': {
        'np_modifiers': ['det', 'amod', 'compound', 'poss', 'nummod'],
        'ignore_np': ['acl:relcl', 'advmod'],
        'vp_aux': ['aux', 'aux:pass', 'compound:prt'],
        'pp_modifiers': ['det', 'amod', 'compound']
    },
    'de': {
        'np_modifiers': ['det', 'amod', 'compound', 'poss', 'nummod'],
        'ignore_np': ['acl:relcl', 'advmod'],
        'vp_aux': ['aux', 'aux:pass', 'compound:prt'],
        'pp_modifiers': ['det', 'amod', 'compound']
    },
    'fr': {
        'np_modifiers': ['det', 'amod', 'compound', 'poss', 'nummod'],
        'ignore_np': ['acl:relcl', 'advmod'],
        'vp_aux': ['aux', 'aux:pass', 'compound:prt'],
        'pp_modifiers': ['det', 'amod', 'compound']
    },
    'it': {
        'np_modifiers': ['det', 'amod', 'compound', 'poss', 'nummod'],
        'ignore_np': ['acl:relcl', 'advmod'],
        'vp_aux': ['aux', 'aux:pass', 'compound:prt'],
        'pp_modifiers': ['det', 'amod', 'compound']
    },
    'pt': {
        'np_modifiers': ['det', 'amod', 'compound', 'poss', 'nummod'],
        'ignore_np': ['acl:relcl', 'advmod'],
        'vp_aux': ['aux', 'aux:pass', 'compound:prt'],
        'pp_modifiers': ['det', 'amod', 'compound']
    },
    'hi': {
        'np_modifiers': ['det', 'amod', 'compound', 'poss', 'nummod'],
        'ignore_np': ['acl:relcl', 'advmod'],
        'vp_aux': ['aux', 'aux:pass', 'compound:prt'],
        'pp_modifiers': ['det', 'amod', 'compound']
    },
    'es': {
        'np_modifiers': ['det', 'amod', 'compound', 'poss', 'nummod'],
        'ignore_np': ['acl:relcl', 'advmod'],
        'vp_aux': ['aux', 'aux:pass', 'compound:prt'],
        'pp_modifiers': ['det', 'amod', 'compound']
    },
    'th': {
        'np_modifiers': ['det', 'amod', 'compound', 'nummod'],
        'ignore_np': ['acl:relcl', 'advmod'],
        'vp_aux': ['aux', 'aux:pass', 'compound:prt'],
        'pp_modifiers': ['det', 'amod', 'compound']
    }
}

# --- English transformation functions ---
def transform_token_english(token, tense, sentence=None):
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
        if feats.get("Tense") == "Past":
            return token["form"]
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
def transform_token_german(token, tense):
    """
    Transforms a German VERB token to the target tense.
      - PRESENT: returns lemma.
      - PAST: returns the token form if marked past; otherwise, naively converts by replacing trailing "en" with "te".
      - FUTURE: returns "wird " + lemma.
    """
    if token["upostag"] != "VERB":
        return token["form"]

    lemma = token.get("lemma", token["form"]).lower()
    feats = token.get("feats", {})

    if tense == "present":
        return lemma
    elif tense == "past":
        if feats.get("Tense") == "Past":
            return token["form"]
        if lemma.endswith("en"):
            return lemma[:-2] + "te"
        else:
            return lemma + "te"
    elif tense == "future":
        return "wird " + lemma
    else:
        return token["form"]

# --- Language configuration ---
LANGUAGE_CONFIG = {
    'en': {
        'transform': transform_token_english
    },
    'de': {
        'transform': transform_token_german
    },
    'fr': {'transform': lambda token, tense: token["form"]},
    'it': {'transform': lambda token, tense: token["form"]},
    'pt': {'transform': lambda token, tense: token["form"]},
    'hi': {'transform': lambda token, tense: token["form"]},
    'es': {'transform': lambda token, tense: token["form"]},
    'th': {'transform': lambda token, tense: token["form"]},
}

def transform_sentence(sentence, lang, tense):
    """
    Transforms a UD sentence into the target tense by modifying VERB tokens using language-specific rules.

    For English in all tenses (present, past, future) auxiliary tokens (upostag "AUX") are filtered out.
    
    Returns:
      - A reconstructed sentence (as a string),
      - The main verb (as identified by a token with deprel "root" and upostag "VERB"),
      - Its index in the sentence.
    """
    transform_func = LANGUAGE_CONFIG[lang]['transform']
    
    # For English, regardless of tense, remove AUX tokens.
    if lang == "en":
        filtered_sentence = [token for token in sentence if token["upostag"] != "AUX"]
    else:
        filtered_sentence = sentence

    new_tokens = []
    main_verb = None
    main_verb_index = None

    for token in filtered_sentence:
        # For VERB tokens, apply the transformation.
        if token["upostag"] == "VERB":
            new_form = transform_func(token, tense, sentence)
        else:
            new_form = token["form"]
        new_tokens.append(new_form)
        # Identify main verb: assume the token with deprel "root" and upostag "VERB" is main.
        # if token.get("deprel") == "root" and token["upostag"] == "VERB" and main_verb is None:
        if token["upostag"] == "VERB" and main_verb is None:
            main_verb = new_form
            main_verb_index = token["id"]

    # Reconstruct the sentence (a simple join with space; for a production version, handle punctuation properly)
    reconstructed = " ".join(new_tokens)
    return reconstructed, main_verb, main_verb_index

def extract_np(tokenlist, head_deprel='nsubj', lang='en'):
    """
    Extract the noun phrase (NP) for a given head token (e.g., subject or object)
    by including its allowed modifiers.
    Returns the NP phrase as a string or None if the head is not found.
    """
    config = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG['en'])
    
    # Identify NP head using the provided dependency relation
    np_head = None
    for token in tokenlist:
        if token.get('deprel') == head_deprel:
            np_head = token
            break
    if np_head is None:
        return None

    np_tokens = [np_head]
    # Collect allowed modifiers attached to NP head
    for token in tokenlist:
        if token.get('head') == np_head['id']:
            dep = token.get('deprel')
            if dep in config['np_modifiers'] and dep not in config['ignore_np']:
                np_tokens.append(token)
    
    np_tokens.sort(key=lambda t: t['id'])
    np_phrase = " ".join(token['form'] for token in np_tokens)
    return np_phrase

def extract_vp(tokenlist, lang='en'):
    """
    Extract the verb phrase (VP) by identifying the main verb (root) and its auxiliaries.
    Returns the VP phrase as a string or None if no main verb is found.
    """
    config = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG['en'])
    main_verb = None
    for token in tokenlist:
        if token.get('deprel') == 'root' and token.get('upos') == 'VERB':
            main_verb = token
            break
    if main_verb is None:
        for token in tokenlist:
            if token.get('upos') == 'VERB':
                main_verb = token
                break
    if main_verb is None:
        return None

    vp_tokens = [main_verb]
    for token in tokenlist:
        if token.get('head') == main_verb['id'] and token.get('deprel') in config['vp_aux']:
            vp_tokens.append(token)
    
    vp_tokens.sort(key=lambda t: t['id'])
    vp_phrase = " ".join(token['form'] for token in vp_tokens)
    return vp_phrase

def extract_obj(tokenlist, lang='en'):
    """
    Extract the object NP using the NP extraction function with head dependency 'obj'.
    """
    return extract_np(tokenlist, head_deprel='obj', lang=lang)

def extract_pp(tokenlist, lang='en'):
    """
    Extract a prepositional phrase (PP) by finding a case marker and the head it relates to,
    along with its modifiers.
    Returns the PP phrase as a string or None if no PP is found.
    """
    config = LANGUAGE_CONFIG.get(lang, LANGUAGE_CONFIG['en'])
    case_token = next((t for t in tokenlist if t.get('deprel') == 'case'), None)
    if case_token is None:
        return None
    pp_head = next((t for t in tokenlist if t.get('id') == case_token.get('head')), None)
    if pp_head is None:
        return None

    pp_tokens = [case_token, pp_head]
    for token in tokenlist:
        if token.get('head') == pp_head['id'] and token.get('deprel') in config['pp_modifiers']:
            pp_tokens.append(token)
    pp_tokens.sort(key=lambda t: t['id'])
    pp_phrase = " ".join(token['form'] for token in pp_tokens)
    return pp_phrase

def extract_svo_sentence(tokenlist, lang='en'):
    """
    Extract and reassemble a simplified SVO sentence by combining:
      - Subject NP,
      - Verb Phrase (VP),
      - Object NP, and
      - Optionally a Prepositional Phrase (PP).
    Returns a controlled sentence string or None if not all components are present.
    """
    subject = extract_np(tokenlist, head_deprel='nsubj', lang=lang)
    verb = extract_vp(tokenlist, lang=lang)
    obj = extract_obj(tokenlist, lang=lang)
    pp = extract_pp(tokenlist, lang=lang)
    
    if subject and verb and obj:
        sentence = f"{subject} {verb} {obj}"
        if pp:
            sentence += f" {pp}"
        return sentence
    return None

def extract_sov_sentence(tokenlist, lang="hi"):
    """
    If the sentence conforms to a simple SOV structure, extract and reassemble
    the subject NP, object NP, and verb phrase VP  (optionally attaching PP to NP).
    
    Returns:
        str: A reassembled sentence (e.g., "The little cat will be eating the fish on the plate"),
             or None if the required elements aren't all found.
    """
    # Use our refined extraction functions.
    subject = extract_np(tokenlist, head_deprel='nsubj', lang=lang)
    verb = extract_vp(tokenlist, lang=lang)
    obj = extract_obj(tokenlist)
    # Optionally, if there is a PP (e.g., adjunct) in the sentence, extract it.
    pp = extract_pp(tokenlist, lang=lang)

    if subject and verb and obj:
        sentence = f"{subject} {obj}"
        if pp:
            sentence += f" {pp}"
        sentence += f" {verb}"
        return sentence
    return None

def prepare_dataset(input_file, output_file, num_sentences=1000, lang="en"):
    """
    Processes a raw UD file by:
      1. Reading the CoNLL-U formatted file.
      2. Extracting simplified SVO sentences using extraction functions.
      3. Annotating each sentence with the language code and the tense information 
         from the main verb (using language-specific future detection).
      4. Writing the processed sentences to a CSV file.
      
    The CSV will contain columns for Language, Tense, Sentence, Main Verb, and Verb Index.
    """
    langs = []
    svo = []
    sov = []

    synthetic_sentences = []
    count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for tokenlist in parse_incr(f):
            if count >= num_sentences:
                break

            # Create three synthetic variants from each UD sentence: present, past, and future.
            for tense in ["present", "past", "future"]:
                sent_text, main_verb, verb_index = transform_sentence(tokenlist, lang, tense)
                synthetic_sentences.append({
                    "language": lang,
                    "tense": tense,
                    "sentence": sent_text,
                    "main_verb": main_verb if main_verb is not None else "",
                    "verb_index": verb_index if main_verb is not None else ""
                })

            processed_count += 1

    with open(input_file, 'r', encoding='utf-8') as f:
        for tokenlist in parse_incr(f):
            # Create synthetic sentence from tokenlist
            if lang in langs:
                if lang in svo:
                    synthetic = extract_svo_sentence(tokenlist, lang=lang)
                elif lang in sov:
                    synthetic = extract_sov_sentence(tokenlist, lang=lang)

                if synthetic:
                    main_verb_token = None
                    verb_index = None
                    for index, token in enumerate(tokenlist):
                        if token.get('upos') == 'VERB':
                            main_verb_token = token
                            verb_index = index
                            break

                    # Get the tense using our heuristic
                    tense = detect_tense(tokenlist, main_verb_token, lang)

                    if tense not in ["Skip", "NA"]:
                        # Extract the full VP phrase, which now includes auxiliaries
                        vp_phrase = extract_vp(tokenlist, lang)
            
                        synthetic_sentences.append({
                            "language": lang,
                            "tense": tense,
                            "sentence": synthetic,
                            "main_verb": vp_phrase,   # Storing the entire VP (e.g., "will be causing")
                            "verb_index": verb_index
                        })
                        count += 1

                    count += 1
                if count >= num_sentences:
                    break
            else:
                print(f"Cannot process {lang}.")
                return

    with open(input_file, 'r', encoding='utf-8') as f:
        for tokenlist in parse_incr(f):
            synthetic = extract_svo_sentence(tokenlist, lang=lang)
            if synthetic is not None:
                # Identify the main verb token (using the 'root' or first VERB token method)
                main_verb_token = None
                verb_index = None
                for index, token in enumerate(tokenlist):
                    if token.get('upos') == 'VERB':
                        main_verb_token = token
                        verb_index = index
                        break

                if main_verb_token is not None:
                    # Get the tense using our heuristic
                    tense = detect_tense(tokenlist, main_verb_token, lang)

                    if tense not in ["Skip", "NA"]:
                        # Extract the full VP phrase, which now includes auxiliaries
                        vp_phrase = extract_vp(tokenlist, lang)
            
                        synthetic_sentences.append({
                            "language": lang,
                            "tense": tense,
                            "sentence": synthetic,
                            "main_verb": vp_phrase,   # Storing the entire VP (e.g., "will be causing")
                            "verb_index": verb_index
                        })
                        count += 1
                else:
                    # If no main verb is found, you might wish to handle the case differently.
                    pass
            if count >= num_sentences:
                break

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["language", "tense", "sentence", "main_verb", "verb_index"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for entry in synthetic_sentences:
            writer.writerow(entry)

    print(f"Saved {len(synthetic_sentences)} synthetic sentences to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Prepare dataset for a specific language.')
    parser.add_argument('--lang', type=str, required=True,
                        help='Language code (e.g., en, de, fr, it, pt, hi, es, th)')
    parser.add_argument('--num_sentences', type=int, default=1000,
                        help='Number of sentences to process')
    args = parser.parse_args()

    print("----------- Processing " + args.lang + " -----------")
    input_file = os.path.join("data", "raw", f"{args.lang}-ud-train.conllu")
    output_file = os.path.join("data", "processed", f"{args.lang}_synthetic.csv")

    prepare_dataset(input_file, output_file, num_sentences=args.num_sentences, lang=args.lang)
