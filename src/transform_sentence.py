from transform_tokens import LANGUAGE_CONFIG

LANGUAGE_CONFIG_SIMPLE = {
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

def get_token_index(token, sentence):
    token_idx = None

    if token is None:
        return None
    token_arr = token.split()
    arr = sentence.split()
    token_idx = arr.index(token_arr[len(token_arr) - 1])

    return token_idx

def extract_np(tokenlist, head_deprel='nsubj', lang='en'):
    """
    Extract the noun phrase (NP) for a given head token (e.g., subject or object)
    by including its allowed modifiers.
    Returns the NP phrase as a string or None if the head is not found.
    """
    config = LANGUAGE_CONFIG_SIMPLE.get(lang, LANGUAGE_CONFIG_SIMPLE['en'])
    
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

def extract_vp(tokenlist, lang='en', tense='present'):
    """
    Extract the verb phrase (VP) by identifying the main verb (root) and its auxiliaries,
    but remove any auxiliary verbs. Then apply the language‐specific tense transformation
    to the VERB tokens.
    Returns the VP phrase as a string.
    """
    config = LANGUAGE_CONFIG_SIMPLE.get(lang, LANGUAGE_CONFIG_SIMPLE['en'])
    main_verb = None
    for token in tokenlist:
        if token.get('upos') == 'VERB':
            main_verb = token
            break
    if main_verb is None:
        return None, None

    # Build VP tokens, skipping any AUX tokens.
    vp_tokens = [main_verb]
    for token in tokenlist:
        # Remove any auxiliary verbs.
        if token.get("upos") == "AUX":
            continue
        if token.get('head') == main_verb['id'] and token.get('deprel') in config['vp_aux']:
            vp_tokens.append(token)
    
    vp_tokens.sort(key=lambda t: t['id'])
    transform_func = LANGUAGE_CONFIG[lang]['transform']
    transformed_vp = []
    for token in vp_tokens:
        new = transform_func(token, tense, tokenlist) or token["form"]
        transformed_vp.append(new)
        # if token["upos"] == "VERB":
        #     transformed_vp.append(transform_func(token, tense, tokenlist))
        # else:
        #     transformed_vp.append(token["form"])

    tmp = transformed_vp[0].split()

    return " ".join(transformed_vp), tmp[len(tmp) - 1]

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
    config = LANGUAGE_CONFIG_SIMPLE.get(lang, LANGUAGE_CONFIG_SIMPLE['en'])
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

def extract_svo_sentence(tokenlist, lang='en', tense='present'):
    """
    Extract and reassemble a simplified SVO sentence by combining:
      - Subject NP,
      - Verb Phrase (VP),
      - Object NP, and
      - Optionally a Prepositional Phrase (PP).
    Returns a controlled sentence string or None if not all components are present.
    """
    subject = extract_np(tokenlist, head_deprel='nsubj', lang=lang)
    verb, main_verb = extract_vp(tokenlist, lang=lang, tense=tense)
    obj = extract_obj(tokenlist, lang=lang)
    pp = extract_pp(tokenlist, lang=lang)
    
    if subject and verb and obj:
        sentence = f"{subject} {verb} {obj}"
        if pp:
            sentence += f" {pp}"

        main_verb_index = get_token_index(main_verb, sentence)
        sentence += " ."
        return sentence, main_verb, main_verb_index
    return None, None, None

def extract_sov_sentence(tokenlist, lang="hi", tense='present'):
    """
    If the sentence conforms to a simple SOV structure, extract and reassemble
    the subject NP, object NP, and verb phrase VP  (optionally attaching PP to NP).
    
    Returns:
        str: A reassembled sentence (e.g., "The little cat will be eating the fish on the plate"),
             or None if the required elements aren't all found.
    """
    # Use our refined extraction functions.
    subject = extract_np(tokenlist, head_deprel='nsubj', lang=lang)
    verb, main_verb = extract_vp(tokenlist, lang=lang, tense=tense)
    obj = extract_obj(tokenlist, lang=lang)
    # Optionally, if there is a PP (e.g., adjunct) in the sentence, extract it.
    pp = extract_pp(tokenlist, lang=lang)

    if subject and verb and obj:
        sentence = f"{subject} {obj}"
        if pp:
            sentence += f" {pp}"
        
        sentence += f" {verb}"
        main_verb_index = get_token_index(main_verb, sentence)
        sentence += " ."
        return sentence, main_verb, main_verb_index
    return None, None, None
