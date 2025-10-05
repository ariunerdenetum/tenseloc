# https://opus.nlpl.eu/OpenSubtitles/en&hi/v2024/OpenSubtitles
# https://huggingface.co/docs/datasets/quickstart#nlp
# https://huggingface.co/datasets/Helsinki-NLP/open_subtitles

from datasets import load_dataset

# This will automatically download the OpenSubtitles data
# dataset = load_dataset("opus", "OpenSubtitles", lang1="en", lang2="hi")
dataset = load_dataset("open_subtitles", lang1="fi", lang2="hi")

# Inspect the splits available (e.g., train, dev, test)
print(dataset)
