import re
import unicodedata


def normalize_text(text: str) -> str:
    if not text:
        return ""

    # remove the diacritic
    text = unicodedata.normalize("NFKD", text.lower().strip())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))

    # if we have 2 or more unused characters -> change it on " "
    text = re.sub(r"[.\-_\xb7\u2022]{2,}", " ", text)

    # remove all punctuations
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)

    # multiple spaces -> one space
    return re.sub(r"\s+", " ", text).strip()


# creating tokens for every not empty word
def tokenize(text: str) -> list[str]:
    return [t for t in normalize_text(text).split() if t]


def strip_leading_number_tokens(tokens: list[str]) -> list[str]:

    # looking for all possible numbers
    roman_num = re.compile(r"^[ivxlcdm]+$", re.IGNORECASE)
    just_num = re.compile(r"^[0-9]+([.][0-9]+)*[.]?$")

    # creating copy of tokens
    result = list(tokens)

    # deleting numbers and leaving only real words
    while result and (just_num.match(result[0]) or roman_num.match(result[0])):
        result.pop(0)
    return result


# creatinfg flat tree of chapters without hierarchy
def flatten_tree(nodes: list, child_key: str = "subchapters") -> list:
    flat = []
    for node in nodes:
        flat.append(node)
        flat.extend(flatten_tree(node.get(child_key, []), child_key))
    return flat
