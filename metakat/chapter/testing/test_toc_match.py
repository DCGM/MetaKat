# import json
# import os
#
# from utils.utils import debugprint

# TODO: export from main properly
# def eval_toc_matches(entries: list[TOCEntry], inputF: str):
#    tp = 0
#    fp = 0
#    fn = 0
#    file_path3 = "./data/grouped_dataZ.json"
#    # Get last foldername from source path and use it as test set
#    current_test_set = os.path.basename(os.path.normpath(inputF))
#
#
#    with open(file_path3, "r", encoding="utf-8") as f:
#        data = json.load(f)
#
#    current_data = data.get(str(current_test_set), None)
#    if current_data is None:
#        print(f"Test set {current_test_set} not found in JSON file.")
#        return
#
#    for match in entries:
#        if not match.chapter_in_text:
#            debugprint("Not found in text => false negative")
#            fn += 1
#            continue
#        if match.filename not in current_data:
#            debugprint("This is not a TOC page => false positive")
#            fp += 1
#            continue
#        if match.chapter_in_text["filename"] not in current_data[match.filename]:
#            debugprint("No match with filename => false positive")
#            fp += 1
#            continue
#        TOCMatch = current_data[match.filename][match.chapter_in_text["filename"]]
#        if TOCMatch > 0:
#            tp += 1
#            current_data[match.filename][match.chapter_in_text["filename"]] -= 1
#        else:
#            debugprint("Found match, but it has no more chapters left")
#            fp += 1
#        if match.subchapters:
#            for subchapter in match.subchapters:
#                if not subchapter.chapter_in_text:
#                    debugprint("Not found in text => false negative")
#                    fn += 1
#                    continue
#                if subchapter.filename not in current_data:
#                    debugprint("This is not a TOC page => false positive")
#                    fp += 1
#                    continue
#                if subchapter.chapter_in_text["filename"] not in current_data[subchapter.filename]:
#                    debugprint("No match with filename => false positive")
#                    fp += 1
#                    continue
#                TOCMatch = current_data[subchapter.filename][subchapter.chapter_in_text["filename"]]
#                if TOCMatch > 0:
#                    tp += 1
#                    current_data[subchapter.filename][subchapter.chapter_in_text["filename"]] -= 1
#                else:
#                    debugprint("Found match, but it has no more chapters left")
#                    fp += 1
#
#    # count not found as separate metric
#    notfound = 0
#    for filename, chapters in current_data.items():
#        for chapter, count in chapters.items():
#            if count > 0:
#                notfound += count
#
#    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#    f1 = 2 * precision * recall / \
#        (precision + recall) if (precision + recall) > 0 else 0.0
#
#    print(f"TP: {tp}")
#    print(f"FP: {fp}")
#    print(f"FN: {fn}")
#    print(f"Precision: {precision:.4f}")
#    print(f"Recall: {recall:.4f}")
#    print(f"F1 Score: {f1:.4f}")
#
#    with open("TOC_matches.json", "a+", encoding="utf-8") as f:
#        json.dump(
#            {
#                "test_set": current_test_set,
#                "TP": tp,
#                "FP": fp,
#                "FN": fn,
#                "Precision": precision,
#                "Recall": recall,
#                "F1 Score": f1,
#                "Undiscovered chapters in text": notfound,
#            },
#            f,
#            indent=4,
#            ensure_ascii=False,
#        )
#        f.write("\n")
#    return tp, fp, fn, precision, recall, f1

# def eval_toc_matches(entries: list[TOCEntry]):
#    tp = 0
#    fp = 0
#    fn = 0
#    file_path3 = "./data/grouped_dataZ.json"
#    # Get last foldername from source path and use it as test set
#    current_test_set = os.path.basename(os.path.normpath(input_folder))
#
#    with open(file_path3, "r", encoding="utf-8") as f:
#        data = json.load(f)
#
#    current_data = data.get(str(current_test_set), None)
#    if current_data is None:
#        print(f"Test set {current_test_set} not found in JSON file.")
#        return
#
#    for match in entries:
#        if not match.chapter_in_text:
#            debugprint("Not found in text => false negative")
#            fn += 1
#            continue
#        if match.filename not in current_data:
#            debugprint("This is not a TOC page => false positive")
#            fp += 1
#            continue
#        if match.chapter_in_text["filename"] not in current_data[match.filename]:
#            debugprint("No match with filename => false positive")
#            fp += 1
#            continue
#        TOCMatch = current_data[match.filename][match.chapter_in_text["filename"]]
#        if TOCMatch > 0:
#            tp += 1
#            current_data[match.filename][match.chapter_in_text["filename"]] -= 1
#        else:
#            debugprint("Found match, but it has no more chapters left")
#            fp += 1
#        if match.subchapters:
#            for subchapter in match.subchapters:
#                if not subchapter.chapter_in_text:
#                    debugprint("Not found in text => false negative")
#                    fn += 1
#                    continue
#                if subchapter.filename not in current_data:
#                    debugprint("This is not a TOC page => false positive")
#                    fp += 1
#                    continue
#                if subchapter.chapter_in_text["filename"] not in current_data[subchapter.filename]:
#                    debugprint("No match with filename => false positive")
#                    fp += 1
#                    continue
#                TOCMatch = current_data[subchapter.filename][subchapter.chapter_in_text["filename"]]
#                if TOCMatch > 0:
#                    tp += 1
#                    current_data[subchapter.filename][subchapter.chapter_in_text["filename"]] -= 1
#                else:
#                    debugprint("Found match, but it has no more chapters left")
#                    fp += 1
#
#    # count not found as separate metric
#    notfound = 0
#    for filename, chapters in current_data.items():
#        for chapter, count in chapters.items():
#            if count > 0:
#                notfound += count
#
#    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
#    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
#    f1 = 2 * precision * recall / \
#        (precision + recall) if (precision + recall) > 0 else 0.0
#
#    print(f"TP: {tp}")
#    print(f"FP: {fp}")
#    print(f"FN: {fn}")
#    print(f"Precision: {precision:.4f}")
#    print(f"Recall: {recall:.4f}")
#    print(f"F1 Score: {f1:.4f}")
#
#    with open("TOC_matches.json", "a+", encoding="utf-8") as f:
#        json.dump(
#            {
#                "test_set": current_test_set,
#                "TP": tp,
#                "FP": fp,
#                "FN": fn,
#                "Precision": precision,
#                "Recall": recall,
#                "F1 Score": f1,
#                "Undiscovered chapters in text": notfound,
#            },
#            f,
#            indent=4,
#            ensure_ascii=False,
#        )
#        f.write("\n")
#    return tp, fp, fn, precision, recall, f1
#
#
# if args.debug:
#    eval_toc_matches(final.TOCStructure)
