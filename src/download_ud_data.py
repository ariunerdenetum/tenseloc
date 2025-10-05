import requests
import os

def download(url, path):
    response = requests.get(url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded: {path}")

if __name__ == "__main__":
    files = {
        # English (UD_English-EWT)
        'data/raw/en-ud-train.conllu': 'https://raw.githubusercontent.com/UniversalDependencies/UD_English-EWT/master/en_ewt-ud-train.conllu',
        # German (UD_German-GSD)
        'data/raw/de-ud-train.conllu': 'https://raw.githubusercontent.com/UniversalDependencies/UD_German-GSD/master/de_gsd-ud-train.conllu',
        # French (UD_French-GSD)
        'data/raw/fr-ud-train.conllu': 'https://raw.githubusercontent.com/UniversalDependencies/UD_French-GSD/master/fr_gsd-ud-train.conllu',
        # Italian (UD_Italian-ISDT)
        'data/raw/it-ud-train.conllu': 'https://raw.githubusercontent.com/UniversalDependencies/UD_Italian-ISDT/master/it_isdt-ud-train.conllu',
        # Portuguese (UD_Portuguese-GSD)
        'data/raw/pt-ud-train.conllu': 'https://raw.githubusercontent.com/UniversalDependencies/UD_Portuguese-GSD/master/pt_gsd-ud-train.conllu',
        # Hindi (UD_Hindi-HDTB)
        'data/raw/hi-ud-train.conllu': 'https://raw.githubusercontent.com/UniversalDependencies/UD_Hindi-HDTB/master/hi_hdtb-ud-train.conllu',
        # Spanish (UD_Spanish-GSD)
        'data/raw/es-ud-train.conllu': 'https://raw.githubusercontent.com/UniversalDependencies/UD_Spanish-GSD/master/es_gsd-ud-train.conllu',
        # Thai (UD_Thai-PUD)
        'data/raw/th-ud-train.conllu': 'https://raw.githubusercontent.com/UniversalDependencies/UD_Thai-PUD/refs/heads/master/th_pud-ud-test.conllu'
    }
    for path, url in files.items():
        download(url, path)
