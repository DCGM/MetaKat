from select import error

import psycopg2
from psycopg2 import Error
import re

pages = "/home/matko/Desktop/images_all.10-4/pages-all"
pages_with_neighbors = "/home/matko/Desktop/images_all.10-4/neighbors/pages_with_neighbors"
neighbors_find = "/home/matko/Desktop/images_all.10-4/neighbors/neighbors_find"
neighbors_cant_find = "/home/matko/Desktop/images_all.10-4/neighbors/neighbors_cant_find"
errors = "/home/matko/Desktop/images_all.10-4/neighbors/errors"

p_w_n_file = open(pages_with_neighbors, "w")
n_find_file = open(neighbors_find, "w")
n_cant_find_file = open(neighbors_cant_find, "w")
errors_file = open(errors, "w")


db_params = {
    'dbname': 'librarymetadata',
    'user': 'librarymetadata',
    'host': 'localhost',
    'port': '5777'
}

conn = psycopg2.connect(**db_params)
cur = conn.cursor()

for line in open(pages):
    img_name = line.strip().split(" ")[0]
    print(img_name)
    clean_id = re.sub(r"^uuid:|^mc_|\.jp.*$", "", img_name)

    p_w_n_file.write("main: " + img_name + "\n")

    try:
        cur.execute("SELECT parent_id, \"order\" FROM meta_records WHERE id = %s", (clean_id,))
        row = cur.fetchone()

        if row is not None:
            parent_id = row[0]
            order = row[1]
        else:
            errors_file.write(img_name + "\n")
            p_w_n_file.write("prev: ERR - main image not in db \n")
            p_w_n_file.write("next: ERR - main image not in db \n")
            continue

        if parent_id is not None and order is not None:
            prev_order = int(order) - 1
            next_order = int(order) + 1

            try:
                cur.execute("SELECT id, \"public\", image_path FROM meta_records WHERE parent_id = %s and \"order\" = %s", (parent_id, prev_order,))
                row = cur.fetchone()

                if row is not None:
                    prev_id = row[0]
                    public = row[1]
                    image_path = row[2]

                    if public is True and image_path is not None:
                        p_w_n_file.write("prev: " + prev_id + "\n")
                        n_find_file.write(prev_id + "\n")
                    else:
                        p_w_n_file.write("prev: " + prev_id + "\n")
                        n_cant_find_file.write(prev_id + "\n")
                else:
                    p_w_n_file.write("prev: None \n")
            except Error as e:
                print(e)
                conn.rollback()

            try:
                cur.execute("SELECT id, \"public\", image_path FROM meta_records WHERE parent_id = %s and \"order\" = %s", (parent_id, next_order,))
                row = cur.fetchone()

                if row is not None:
                    next_id = row[0]
                    public = row[1]
                    image_path = row[2]

                    if public is True and image_path is not None:
                        p_w_n_file.write("next: " + next_id + "\n")
                        n_find_file.write(next_id + "\n")
                    else:
                        p_w_n_file.write("next: " + next_id + "\n")
                        n_cant_find_file.write(next_id + "\n")
                else:
                    p_w_n_file.write("next: None \n")
            except Error as e:
                print(e)
                conn.rollback()

        else:
            p_w_n_file.write("prev: ERR - No parent id or order \n")
            p_w_n_file.write("next: ERR - No parent id or order \n")
    except Error as e:
        print(e)
        errors_file.write(img_name + "\n")
        p_w_n_file.write("prev: ERR - main image search fail \n")
        p_w_n_file.write("next: ERR - main image search fail \n")
        conn.rollback()

p_w_n_file.close()
n_find_file.close()
n_cant_find_file.close()
errors_file.close()

cur.close()
conn.close()
