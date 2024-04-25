import torch
from fuzzywuzzy import fuzz

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
