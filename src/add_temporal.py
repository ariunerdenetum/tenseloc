import pandas as pd
import random

# 1. Load CSV and filter for English
data_dir = './data/processed/'
df = pd.read_csv(data_dir + 'all_sentences_train.csv')
df_en = df[df['language'] == 'en']

# 2. Define tenses and temporal phrases
tenses = ['past', 'present', 'future']
temporal_phrases = {
    'past': [
        'Yesterday', 'Last night', 'Earlier today', 'A week ago',
        'In my childhood', 'Previously', 'Back then', 'Last Monday',
        'Last summer', 'Once upon a time'
    ],
    'present': [
        'Everyday', 'Always', 'Often', 'Usually',
        'Every week', 'Sometimes', 'Every saturday', 'Rarely',
        'Every monday', 'Every morning'
    ],
    'future': [
        'Tomorrow', 'Next week', 'Soon', 'In the near future',
        'By next year', 'Later today', 'In a few days', 'Shortly',
        'Before long', 'Next year'
    ]
}

# 3. Sample first 500 sentences per tense
samples = []
for tense in tenses:
    sub = df_en[df_en['tense'] == tense].head(500).copy()
    sub.reset_index(drop=True, inplace=True)
    phrases = temporal_phrases[tense]
    n_phrases = len(phrases)
    augmented_rows = []

    for i, row in sub.iterrows():
        base_sentence = row['sentence']
        base = base_sentence[0].lower() + base_sentence[1:]
        phrase = random.choice(phrases)
        phrase_length = len(phrase.split())
        new_sentence = f"{phrase} {base}"
        new_index = row['verb_index'] + phrase_length
        phrase_index = phrase_length - 1
        augmented_rows.append({
            'language': 'en',
            'tense': tense,
            'sentence': new_sentence,
            'main_verb': row['main_verb'],
            'verb_index': new_index,
        })

    samples.append(pd.DataFrame(augmented_rows))

# 4. Concatenate all augmented dataframes
augmented_df = pd.concat(samples, ignore_index=True)

# 5. Save to CSV
augmented_df.to_csv(data_dir + 'all_sentences_temporal.csv', index=False)
