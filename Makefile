.PHONY: download_ud_data, download_opus_data, preprocess_data, merge_sentences, translate, add_temporal, generate_prompt_en, generate_prompt_de

NUM_SENTENCES = 1000

download_ud_data:
	python3 src/download_ud_data.py

download_opus_data:
	python3 src/download_opus_data.py

preprocess_data:
	python3 src/preprocess_data_v2.py --lang $(lang) --num_sentences $(NUM_SENTENCES)

merge_sentences:
	python3 src/merge_sentences.py

translate:
	python3 src/translation.py

add_temporal:
	python3 src/add_temporal.py

generate_prompt_en:
	python3 src/generate_prompt_en.py

generate_prompt_de:
	python3 src/generate_prompt_de.py

# E.g.,
# input=data/processed/en_synthetic.csv
# output=data/processed/en_global.csv
