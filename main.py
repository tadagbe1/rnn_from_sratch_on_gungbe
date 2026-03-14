from rnn.rnn import RNN
import unicodedata

def load_data(data_path):
    data = ''
    with open(data_path, 'r', encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data += line
    return data

def get_vocab(data):
    return list(set(list(data)))

def normalize(data):
    return unicodedata.normalize('NFC', data)


def clean_text(text):
    text = unicodedata.normalize("NFC", text)

    allowed_extra = set(
        "\n\t "
        ".,;:!?\"'()[]{}-–—/@&%+$=«»"
    )

    cleaned = []

    for ch in text:
        cat = unicodedata.category(ch)

        # garder lettres et chiffres latins étendus
        if cat.startswith("L") or cat.startswith("N"):
            name = unicodedata.name(ch, "")
            if "LATIN" in name:
                cleaned.append(ch)

        # garder ponctuation / espaces choisis
        elif ch in allowed_extra:
            cleaned.append(ch)

    data = "".join(cleaned)
    data.replace('\n', '')

    return "".join(cleaned)



if __name__ == "__main__":
    data_path = './data/afranaph.txt'
    text = load_data(data_path)
    data = normalize(text).lower()
    
    vocab = sorted(list(set(data)))
    print(vocab)
    print("Corpus length:", len(data))
    print("Vocab size:", len(vocab))
    rnn = RNN(vocab)
    rnn.train(data)

