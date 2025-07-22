import os, random, re

descriptions = "/home/matko/Desktop/type-descriptions"
all_pages = "/home/matko/Desktop/images_all.10-4/pages.trn.out"
out = "/home/matko/Desktop/training_descriptions"

with open(all_pages, "r") as pages:
    for i, page in enumerate(pages):
        p_id = page.strip().split()[0]
        p_type = page.strip().split()[1]

        clean_id = re.sub(r"\.jp.*$", "", p_id)

        type_file = os.path.join(descriptions, p_type)

        with open(type_file, "r") as f:
            lines = f.readlines()
            random_line = random.choice(lines)

            out_file = os.path.join(out, clean_id + ".txt")

            with open(out_file, "w") as out_f:
                out_f.write(random_line)