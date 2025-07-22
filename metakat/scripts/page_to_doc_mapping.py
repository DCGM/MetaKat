import psycopg2
from psycopg2 import Error
import re

pages = "/home/matko/Desktop/new_images/new_images.txt"
doc = "/home/matko/Desktop/new_images/page_to_doc_mapping"

page_file = open(pages, "r")
doc_file = open(doc, "w")

db_params = {
    'dbname': 'librarymetadata',
    'user': 'librarymetadata',
    'host': 'localhost',
    'port': '5777'
}

conn = psycopg2.connect(**db_params)
cur = conn.cursor()

for line in page_file:
    p_id = line.strip().split()[0]
    clean_id = re.sub(r"^uuid:|^mc_|\.jp.*$", "", p_id)

    try:
        cur.execute("SELECT parent_id FROM meta_records WHERE id = %s", (clean_id,))
        row = cur.fetchone()

        if row is not None:
            parent_id = row[0]

            doc_file.write(f"{clean_id} {parent_id} \n")
        else:
            doc_file.write(f"{clean_id} ERR - nothing found \n")
    except Error as e:
        print(e)
        doc_file.write(f"{clean_id} ERR - database error \n")
        cur.rollback()

page_file.close()
doc_file.close()