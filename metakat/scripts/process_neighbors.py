import os

from PIL import Image

neighbors = "/home/matko/Desktop/images_all.10-4/neighbors/pages_with_neighbors"
all_images = "/home/matko/Desktop/neighbors-all/images"
act_neighbors = "/home/matko/Desktop/actual_neighbors/neighbors"
act_images = "/home/matko/Desktop/actual_neighbors/images"

act_file = open(act_neighbors, "w")

for i, line in enumerate(open(neighbors, "r")):
    if (i + 1) % 1000 == 0:
        print(f"processed {i + 1} images")

    pos, img_id = line.strip().split(": ")

    if pos == 'main':
        act_file.write(line.strip() + "\n")
    else:
        if 'ERR'in img_id or 'None' in img_id:
            act_file.write(line.strip() + "\n")
            continue
        else:
            found = False
            for img_name in os.listdir(all_images):
                if img_id in img_name:
                    found = True
                    break

            try:
                image = Image.open(all_images + "/" + img_name)
            except Exception as e:
                print(e)
                found=False

            if found:
                act_file.write(pos + ": " + img_name + "\n")
                image.save(act_images + "/" + img_name)
            else:
                act_file.write(pos + ": ERR - not in image folder\n")


act_file.close()