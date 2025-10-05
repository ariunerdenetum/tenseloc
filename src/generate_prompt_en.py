import random
import csv
import json
from pattern.en import conjugate, lemma, PRESENT, PAST, FUTURE


# Signal words per tense
signal_words = {
    'past': [
        'Yesterday', 'Last night', 'Earlier today', 'A week ago',
        'In my childhood', 'Previously', 'Back then', 'Last Monday',
        'Last summer', 'Once upon a time'
    ],
    'present': [
        'Everyday', 'Always', 'Often', 'Usually',
        'Every week', 'Sometimes', 'Every Saturday', 'In this instant',
        'Every Monday', 'Every morning'
    ],
    'future': [
        'Tomorrow', 'Next week', 'Soon', 'In the near future',
        'By next year', 'Later today', 'In a few days', 'Shortly',
        'Before long', 'Next year'
    ]
}

# Subjects and objects
subjects = [
    'I', 'you', 'he', 'she', 'we', 'they',
    'the dog', 'the cat', 'the birds', 'the teacher', 'the student',
    'the doctor', 'the engineer', 'the artist', 'the musician',
    'the company', 'the team', 'the robot', 'the child', 'the parent',
    'the driver', 'the cooks', 'the athlete'
]
objects = [
    'the mailman', 'a letter', 'the ball', 'the homework', 'the song',
    'the movie', 'the problem', 'the book', 'the cake', 'the test',
    'the meeting', 'the presentation', 'the game', 'the car', 'the house',
    'the garden', 'the computer', 'the phone', 'the cake', 'the dinner',
    'the project', 'the lecture', 'the exam', 'the journal', 'the painting'
]

# Main verbs list expanded
main_verbs = [
    'smile', 'laugh', 'cry', 'bark', 'meow', 'chirp', 'walk', 'jog',
    'write', 'speak', 'eat', 'drink', 'play', 'watch', 'listen', 'dance',
    'sing', 'draw', 'paint', 'cook', 'bake', 'drive', 'fly', 'jump', 'sleep',
    'think', 'dream', 'build', 'solve', 'teach', 'learn', 'create', 'deliver',
    'design', 'explore', 'discover', 'see', 'carry', 'like', 'love', 'seek',
    'summarize', 'construct'
]

# Determine grammatical person and number based on subject
def get_person_number(subject):
    s = subject.lower()
    if s == 'i': return 1, 'singular'
    if s == 'we': return 1, 'plural'
    if s == 'you': return 2, 'singular'
    if s == 'they': return 3, 'plural'
    if s in ('he', 'she', 'it') or s.startswith('the '): return 3, 'singular'
    return 3, 'singular'

# Template builder with structured record
def build_prompt_record(record_id, signal, subj, obj, forms, correct_label):
    # Shuffle options
    items = list(forms.items())  # [('lemma', 'smile'), ...]
    random.shuffle(items)
    labels = ['A', 'B', 'C']
    options = {}
    tense_mapping = {}
    gold_answer = None
    
    for lab, (form_name, verb) in zip(labels, items):
        options[lab] = form_name  # Store tense label
        tense_mapping[lab] = form_name
        if form_name == correct_label:
            gold_answer = lab

    prompt_text_lines = [f"{signal}, {subj} ___ {obj}."]
    # for lab in labels:
    #     prompt_text_lines.append(f"{lab}) {options[lab]}")
    # prompt_text = '\n'.join(prompt_text_lines)

    for lab in labels:
        display = forms[tense_mapping[lab]]
        prompt_text_lines.append(f"{lab}) {display}")
    prompt_text_lines.append("Answer:")
    prompt_text = '\n'.join(prompt_text_lines)

    return {
        'prompt_id': record_id,
        'gold_tense': correct_label,
        'gold_answer': gold_answer,
        # 'signal_word': signal,
        # 'subject': subj,
        # 'object': obj,
        'option_A': options['A'],
        'option_B': options['B'],
        'option_C': options['C'],
        'prompt_text': prompt_text
    }

# Generation function
def generate_prompt_records(n_per_tense=300, seed=42):
    random.seed(seed)
    records = []
    record_id = 1
    for tense in ['past', 'present', 'future']:
        for _ in range(n_per_tense):
            signal = random.choice(signal_words[tense])
            subj = random.choice(subjects)
            obj = random.choice(objects)
            verb = random.choice(main_verbs)
            # Prepare conjugations
            lemma_verb = lemma(verb)
            person, number = get_person_number(subj)
            verb_pres = conjugate(verb, tense=PRESENT, person=person, number=number)
            verb_past = conjugate(verb, tense=PAST)
            verb_fut = f"will {lemma_verb}"

            forms = {
                'present': verb_pres,
                'past': verb_past,
                'future': verb_fut
            }

            record = build_prompt_record(record_id, signal, subj, obj, forms, tense)
            records.append(record)
            record_id += 1
    return records

# Save to CSV and JSON
def main():
    records = generate_prompt_records()
    # fieldnames = ['prompt_id','gold_tense','gold_answer','signal_word','subject','object','option_A','option_B','option_C','option_D','prompt_text']
    fieldnames = ['prompt_id','gold_tense','gold_answer','option_A','option_B','option_C','prompt_text']
    # with open('./data/processed/prompts_dataset_test.csv', 'w', newline='', encoding='utf-8') as csvfile:
    #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #     writer.writeheader()
    #     writer.writerows(records)
    with open('./data/processed/prompts_dataset_test.json', 'w', encoding='utf-8') as jsonfile:
        json.dump(records, jsonfile, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    main()
