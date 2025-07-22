import re, os

all_pages = "/home/matko/Desktop/bakalarka/data/ann/pages.trn"
out = "/home/matko/Desktop/normal_descriptions"

with open(all_pages, "r") as pages:
    for i, page in enumerate(pages):
        p_id = page.strip().split()[0]
        p_type = page.strip().split()[1]

        clean_id = re.sub(r"\.jp.*$", "", p_id)
        split_type = re.findall(r'[A-Z][^A-Z]*', p_type)
        split_type = ' '.join(filter(None, split_type))


        out_file = os.path.join(out, clean_id + ".txt")

        with open(out_file, "w") as out_f:
            out_f.write("This is a page of type " + split_type)