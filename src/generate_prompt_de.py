import random
import json
from pattern.de import conjugate, lemma, PRESENT, PAST, FUTURE

# Signal words per tense (German)
signal_words = {
    'past': [
        'Gestern', 'Letzte Nacht', 'Vor kurzem', 'Vor einer Woche',
        'Früher', 'Damals', 'Vor Jahren', 'Letzten Montag',
        'Letzten Sommer', 'Einst'
    ],
    'present': [
        'Jeden Tag', 'Immer', 'Oft', 'Meistens',
        'Manchmal', 'Jede Woche', 'Jeden Samstag', 'Jetzt',
        'Montags', 'Morgens'
    ],
    'future': [
        'Morgen', 'Nächste Woche', 'Bald', 'In naher Zukunft', 
        'Später', 'In ein paar Tagen', 'Demnächst',
        'Bevor es zu spät ist', 'Im nächsten Jahr'
    ]
}

# Subjects (German pronouns and noun phrases)
subjects = [
    'ich', 'du', 'er', 'sie', 'wir', 'ihr', 'sie',
    'der Hund', 'die Katze', 'die Vögel', 'der Lehrer', 'die Studentin',
    'der Arzt', 'der Ingenieur', 'der Künstler', 'die Musikerin',
    'die Firma', 'das Team', 'der Roboter', 'das Kind', 'der Elternteil'
]

# Objects (German noun phrases)
objects = [
    'den Briefträger', 'einen Brief', 'den Ball', 'die Hausaufgabe',
    'das Lied', 'den Film', 'das Problem', 'das Buch', 'den Kuchen',
    'den Test', 'die Besprechung', 'die Präsentation', 'das Spiel',
    'das Auto', 'das Haus', 'den Garten', 'den Computer', 'das Telefon',
    'das Projekt', 'die Vorlesung', 'die Prüfung', 'das Tagebuch', 'das Gemälde'
]

# Main verbs list in German
main_verbs = [
    'lächeln', 'lachen', 'weinen', 'bellen', 'miauen', 'zwitschern',
    'gehen', 'laufen', 'schreiben', 'sprechen', 'essen', 'trinken',
    'spielen', 'sehen', 'hören', 'tanzen', 'singen', 'zeichnen', 'malen',
    'kochen', 'backen', 'fahren', 'fliegen', 'springen', 'schlafen',
    'denken', 'träumen', 'bauen', 'lösen', 'unterrichten', 'lernen',
    'erschaffen', 'liefern', 'gestalten', 'erkunden', 'entdecken', 'tragen',
    'mögen', 'lieben', 'suchen', 'zusammenfassen', 'konstruieren'
]

# Determine grammatical person and number based on subject
def get_person_number(subject):
    s = subject.lower()
    if s == 'ich': return 1, 'singular'
    if s == 'wir': return 1, 'plural'
    if s == 'du': return 2, 'singular'
    if s == 'ihr': return 2, 'plural'
    if s in ('er', 'sie', 'es') or s.startswith('der ') or s.startswith('die ') or s.startswith('das '):
        # singular nouns
        return 3, 'singular'
    return 3, 'plural'

# Build a single prompt record
def build_prompt_record(record_id, signal, subj, obj, forms, correct_label):
    items = list(forms.items())
    random.shuffle(items)
    labels = ['A', 'B', 'C']
    options = {}
    gold_answer = None

    for lab, (form_name, verb_form) in zip(labels, items):
        options[lab] = verb_form
        if form_name == correct_label:
            gold_answer = lab

    prompt_lines = [f"{signal}, {subj} ___ {obj}."]
    for lab in labels:
        prompt_lines.append(f"{lab}) {options[lab]}")
    prompt_lines.append("Antwort:")
    prompt_text = '\n'.join(prompt_lines)

    return {
        'prompt_id': record_id,
        'gold_tense': correct_label,
        'gold_answer': gold_answer,
        'option_A': options['A'],
        'option_B': options['B'],
        'option_C': options['C'],
        'prompt_text': prompt_text
    }

# Generate dataset records
def generate_prompt_records(n_per_tense=30, seed=42):
    random.seed(seed)
    records = []
    record_id = 1
    for tense in ['past', 'present', 'future']:
        for _ in range(n_per_tense):
            signal = random.choice(signal_words[tense])
            subj = random.choice(subjects)
            obj = random.choice(objects)
            verb = random.choice(main_verbs)

            lemma_verb = lemma(verb)
            person, number = get_person_number(subj)
            verb_pres = conjugate(verb, tense=PRESENT, person=person, number=number)
            verb_past = conjugate(verb, tense=PAST, person=person, number=number)
            verb_fut = f"{conjugate('werden', tense=PRESENT, person=person, number=number)} {lemma_verb}"

            forms = {
                'present': verb_pres,
                'past':    verb_past,
                'future':  verb_fut
            }

            record = build_prompt_record(record_id, signal, subj, obj, forms, tense)
            records.append(record)
            record_id += 1
    return records

# Main: write JSON
if __name__ == '__main__':
    records = generate_prompt_records()
    with open('./data/processed/prompts_dataset_dev_de.json', 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
