import torch
from fuzzywuzzy import fuzz
import Levenshtein

def ner_pipeline(text, tokenizer, model, device):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = torch.argmax(logits, dim=2)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    results = []
    for token, prediction in zip(tokens, predictions[0]):
        results.append([token, model.config.id2label[prediction.item()]])
    return results


def connect_words(ner_output):
    result = []
    for i in range(len(ner_output)):
        has_hashtag = ner_output[i][0].startswith("##")
        ner_output[i][0] = ner_output[i][0].replace("##", "")
        if ner_output[i][1].startswith("B-"):
            if has_hashtag and ner_output[i][1].endswith("T"):
                if len(result) > 0:
                    result[-1][0] += ner_output[i][0]
                    continue                    
            result.append([ner_output[i][0], ner_output[i][1][-1]])
        elif ner_output[i][1].startswith("I-"):
            if len(result) > 0:
                result[-1][0] += " " + ner_output[i][0]
            else:
                result.append([ner_output[i][0], ner_output[i][1][-1]])
    return result


def remove_special_tokens(ner_output):
    result = []
    for i in range(len(ner_output)):
        if ner_output[i][0] != "[CLS]" and ner_output[i][0] != "[SEP]":
            result.append(ner_output[i])
    return result

def find_matching_text(original_text, ner_text):
    original_words = original_text.split()

    best_match_start = 0
    best_match_end = 0
    best_ratio = 0

    for i in range(len(original_words)):
        for j in range(i, len(original_words)):
            substring = ' '.join(original_words[i:j+1])
            ratio = fuzz.ratio(substring, ner_text)
            if ratio > best_ratio:
                best_ratio = ratio
                best_match_start = i
                best_match_end = j

    best_match_text = ' '.join(original_words[best_match_start:best_match_end+1])
    return best_match_text


def correct_spacing(ner_output, text):
    for i in range(len(ner_output)):
        ner_text = ner_output[i][0]
        if ner_text not in text:
            ner_text = find_matching_text(text, ner_text)
        if ner_text in text:
            ner_output[i][0] = ner_text
        else:
            print(ner_text + " not in text")
    return ner_output

def is_roman_number(s):
    s = s.replace(" ", "").replace(".", "").replace(",", "").replace(";", "").replace(":", "")
    roman_numbers = ["I", "V", "X", "L", "C", "D", "M"]
    for c in s:
        if c not in roman_numbers:
            return False
    return True

def dict_matching(line_transcription):
    line_transcription = line_transcription.split()
    to_match_dict = {
        "redaktor": "REDAKTOR",
        "redaktorem": "REDAKTOR",
        "redaktoři": "REDAKTOR",
        "rediguje": "REDAKTOR",
        "redakce": "REDAKTOR",
        "nakladatel": "NAKLADATEL",
        "nakladatelem": "NAKLADATEL",
        "nakladatelé": "NAKLADATEL",
        "nakladatelství": "NAKLADATEL",
        "nakládá": "NAKLADATEL",
        "vydavatel": "VYDAVATEL",
        "vydavatelem": "VYDAVATEL",
        "vydavatelé": "VYDAVATEL",
        "vydavatelství": "VYDAVATEL",
        "vydává": "VYDAVATEL",
        "wydawatel": "VYDAVATEL",
        "ročník": "ROČNÍK",
        "ročníku": "ROČNÍK",
        "číslo": "ČÍSLO",
        "sešit": "ČÍSLO",
    }
    
    to_exact_match = {
        "č.": "ČÍSLO",
        "č": "ČÍSLO",
    }

    matched = []
    for transcription_word in line_transcription:
        word_match = []
        if is_roman_number(transcription_word):
            matched.append([transcription_word, "ŘÍMSKÉ ČÍSLO"])
        elif transcription_word in to_exact_match:
            matched.append([transcription_word, to_exact_match[transcription_word]])
        else:
            transcription_word = transcription_word.replace(",", "").replace(".", "").replace(";", "").replace(":", "").lower()
            for to_match in to_match_dict.items():
                l_distance = Levenshtein.distance(transcription_word, to_match[0])
                if l_distance <= 2:
                    word_match.append([transcription_word, to_match_dict[to_match[0]], l_distance])
            if len(word_match) > 0:
                word_match = sorted(word_match, key=lambda x: x[2])      
                matched.append([word_match[0][0], word_match[0][1]])
                
    return matched
