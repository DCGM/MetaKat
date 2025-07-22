import os

import psycopg2
from psycopg2 import Error
import re

max_types_per_parent = 5
limit = 500

folder = "/home/matko/Desktop/new_pages"
os.makedirs(folder, exist_ok=True)

page_file = os.path.join(folder, "pages.2024-11-04.all")
page_to_doc = os.path.join(folder, "page_to_doc.txt")
pages = os.path.join(folder, "pages.txt")
cant_find = os.path.join(folder, "cant_find.txt")
neighbors = os.path.join(folder, "neighbors.txt")

db_params = {
    'dbname': 'librarymetadata',
    'user': 'librarymetadata',
    'host': 'localhost',
    'port': '5777'
}

sql_querry = """
    SELECT *
    FROM (
        SELECT id, \"public\", image_path, parent_id, page_type,
               ROW_NUMBER() OVER (PARTITION BY parent_id ORDER BY "id") as rn
        FROM meta_records
        WHERE page_type = %s
    ) sub
    WHERE rn <= %s LIMIT %s;
"""

conn = psycopg2.connect(**db_params)
cur = conn.cursor()

classes = ('Abstract,Advertisement,Appendix,BackCover,BackEndPaper,BackEndSheet,Bibliography,'
                     'Blank,CalibrationTable,Cover,CustomInclude,Dedication,Edge,Errata,FlyLeaf,'
                     'FragmentsOfBookbinding,FrontCover,FrontEndPaper,FrontEndSheet,FrontJacket,'
                     'Frontispiece,Illustration,Impressum,Imprimatur,Index,Jacket,ListOfIllustrations,'
                     'ListOfMaps,ListOfTables,Map,NormalPage,Obituary,Preface,SheetMusic,Spine,Table,'
                     'TableOfContents,TitlePage')
classes = classes.split(",")

for page_type in classes:
    try:
        cur.execute(sql_querry, (page_type, max_types_per_parent, limit))
        result = cur.fetchall()

        if result:
            for row in result:
                page_id = row[0]
                public = row[1]
                image_path = row[2]
                parent_id = row[3]
                p_type = row[4]

                found = False

                with open(page_file, 'r+') as f:
                    for line in f:
                        if page_id in line:
                            found = True
                            break

                if found is False:
                    with open(neighbors, 'a') as f:
                        f.write(page_id + "\n")
                    with open(page_file, 'a') as f:
                        f.write(page_id + " " + p_type + "\n")

                    with open(page_to_doc, 'a') as f:
                        f.write(page_id + " " + parent_id + "\n")

                    if public is True and image_path is not None:
                        with open(pages, 'a') as f:
                            f.write(page_id + "\n")
                    else:
                        with open(cant_find, 'a') as f:
                            f.write(page_id + "\n")
        else:
            print("Nothing found")
    except Error as e:
        print(e)
        conn.rollback()

cur.close()
conn.close()