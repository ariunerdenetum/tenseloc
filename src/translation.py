# pip install deep-translator
from deep_translator import GoogleTranslator
import json

# 1/ Define English templates
english_prompts = [
    # (id, prompt_block)
    ("pas_en1", "drank",   "I sat on the rug. I drank tea.\n"
                    "I sat on the rug. I"),
    ("pas_en2", "ate",   "Lily the cat relaxed on the mat. she ate an apple.\n"
                    "Lily the cat relaxed on the mat. she"),
    ("pas_en3", "ran",   "Aki the dog barked at the mailman. he ran away.\n"
                    "Aki the dog barked at the mailman. he"),
    ("pas_en4", "chirped",   "The birds perched on the branch. they chirped a tune.\n"
                    "The birds perched on the branch. they"),
    ("pas_en5", "talked",   "We sat by the fire. we talked softly.\n"
                    "We sat by the fire. we"),
    ("pre_en1", "drink",   "I sit on the rug. I drink tea.\n"
                    "I sit on the rug. I"),
    ("pre_en2", "eats",   "Lily the cat relaxes on the mat. she eats an apple.\n"
                    "Lily the cat relaxes on the mat. she"),
    ("pre_en3", "runs",   "Aki the dog barks at the mailman. he runs away.\n"
                    "Aki the dog barks at the mailman. he"),
    ("pre_en4", "chirp",   "The birds perch on the branch. they chirp a tune.\n"
                    "The birds perch on the branch. they"),
    ("pre_en5", "talk",   "We sit by the fire. we talk softly.\n"
                    "We sit by the fire. we"),
    ("fut_en1", "drink",   "I'll sit on the rug. I'll drink tea.\n"
                    "I'll sit on the rug. I"),
    ("fut_en2", "will",   "Lily the cat'll relax on the mat. she'll eat an apple.\n"
                    "Lily the cat'll relax on the mat. she"),
    ("fut_en3", "will",    "Aki the dog'll bark at the mailman. he'll run away.\n"
                    "Aki the dog'll bark at the mailman. he"),
    ("fut_en4", "will",    "The birds'll perch on the branch. they'll chirp a tune.\n"
                    "The birds'll perch on the branch. they"),
    ("fut_en5", "will",    "We'll sit by the fire. we'll talk softly.\n"
                    "We'll sit by the fire. we"),
]

languages = {
    "de": "German",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "es": "Spanish",
    "hi": "Hindi",
    "th": "Thai"
}

all_translated = []

for subject_key, gold, eng_block in english_prompts:
    all_translated.append({
        "prompt_id": subject_key,
        "gold": gold,
        "words_restore": [],
        "prompt": eng_block
    })

for lang_code, lang_name in languages.items():
    # all_translated[lang_name] = []
    for subject_key, gold, eng_block in english_prompts:
        lines = eng_block.split("\n")
        trans_lines = []
        for line in lines:
            if not line.strip():
                continue
            # synchronous translate
            translated = GoogleTranslator(source='en', target=lang_code).translate(line)
            trans_lines.append(translated)
        prompt_block = "\n".join(trans_lines)
        new_key = subject_key[:4] + lang_code + subject_key[6:]
        # all_translated[lang_name].append((new_key, gold, prompt_block))
        all_translated.append({
            "prompt_id": new_key,
            "gold": "",
            "words_restore": [],
            "prompt": prompt_block
        })

# Export to JSON for manual verification
with open("translated_prompts.json", "w", encoding="utf8") as f:
    json.dump(all_translated, f, ensure_ascii=False, indent=2)
