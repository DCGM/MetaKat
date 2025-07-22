import os
from binascii import Error

from PIL import Image

folder_path = "/home/matko/Desktop/new_images/images_mix"
images = "/home/matko/Desktop/new_images/new_images"
all_text = "/home/matko/Desktop/new_images/corect_main.txt"
new_text = "/home/matko/Desktop/new_images/new_images.txt"

for i, filename in enumerate(os.listdir(folder_path)):

    try:
        image = Image.open(folder_path + "/" + filename)
    except Error as e:
        pass

    save_line = ""
    for line in open(all_text, 'r'):
        save_line = line
        img_id = line.strip().split(" ")[0]

        if img_id == filename:
            with open(new_text, 'a') as f:
                f.write(save_line)

            image.save(images + "/" + filename)