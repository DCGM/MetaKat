import psycopg2
from psycopg2 import Error
import re

main = "/home/matko/Desktop/images_all.10-4/neighbors/neighbors_find"
# correct_main = "/home/matko/Desktop/new_images/corect_main.txt"
public = "/home/matko/Desktop/neighbors_image_path"
# not_public = "/home/matko/Desktop/new_images/not_public.txt"

# correct_main_file = open(correct_main, 'w')
public_file = open(public, "w")
# not_public_file = open(not_public, "w")

db_params = {
    'dbname': 'librarymetadata',
    'user': 'librarymetadata',
    'host': 'localhost',
    'port': '5777'
}

conn = psycopg2.connect(**db_params)
cur = conn.cursor()

for i, line in enumerate(open(main)):
    img_name = line.strip()
    # img_name = line.strip().split(" ")[0]
    # page_type = line.strip().split(" ")[1]

    # correct_main_file.write(img_name + ".jpg " + page_type + "\n")

    clean_id = re.sub(r"^uuid:|^mc_|\.jp.*$", "", img_name)

    if (i + 1) % 1000 == 0:
        print(f'Processed images: {i + 1}')

    try:
        cur.execute("SELECT \"public\", image_path FROM meta_records WHERE id = %s", (clean_id,))
        row = cur.fetchone()

        if row is not None:
            pub = row[0]
            path = row[1]

            if pub is True and path is not None:
                clean_path = path.replace("uuid:", '')
                public_file.write(clean_path + "\n")
            else:
                # not_public_file.write(clean_id + " NOT PUBLIC\n")
                pass

        else:
            print(f"{clean_id} NOT FOUND\n")

    except Error as e:
        print(e)
        conn.rollback()

# correct_main_file.close()
public_file.close()
# not_public_file.close()

cur.close()
conn.close()

