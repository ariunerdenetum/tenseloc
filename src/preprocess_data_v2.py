import os
import csv
import argparse
from conllu import parse_incr
from transform_sentence import extract_svo_sentence, extract_sov_sentence


def transform_sentence(tokenlist, lang, tense):
    """
    Transforms a UD sentence into a simplified, short sentence using an SVO extraction
    (or SOV for Hindi) and applies tense manipulation to the VP.
    If extraction is successful, returns the simplified sentence; otherwise, falls back to token-by-token transformation.
    """
    if lang in ['en', 'de', 'fr', 'it', 'pt', 'es', 'th']:
        simple_sentence, main_verb, main_verb_index = extract_svo_sentence(tokenlist, lang, tense)
    elif lang in ['hi']:
        simple_sentence, main_verb, main_verb_index = extract_sov_sentence(tokenlist, lang, tense)
    else:
        simple_sentence = None

    if simple_sentence is not None:
        return simple_sentence, main_verb, main_verb_index
    else:
        return None, None, None

def prepare_dataset(input_file, output_file, num_sentences=1000, lang="en"):
    processed_count = 0

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(input_file, "r", encoding="utf-8") as conllu_file, \
         open(output_file, "w", newline="", encoding="utf-8") as out_csv:
        fieldnames = ["language", "tense", "sentence", "main_verb", "verb_index"]
        writer = csv.DictWriter(out_csv, fieldnames=fieldnames)
        writer.writeheader()
        
        
        for tokenlist in parse_incr(conllu_file):
            if processed_count >= num_sentences:
                break

            success = True
            # Create three synthetic variants from each UD sentence: present, past, and future.
            for tense in ["present", "past", "future"]:
                sent_text, main_verb, verb_index = transform_sentence(tokenlist, lang, tense)

                if not sent_text:
                    success = False
                    break

                writer.writerow({
                    "language": lang,
                    "tense": tense,
                    "sentence": sent_text,
                    "main_verb": main_verb if main_verb is not None else "",
                    "verb_index": verb_index if main_verb is not None else ""
                })
                
            if success:
                processed_count += 1

    print(f"Saved {processed_count} synthetic sentences to {output_file}")

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
