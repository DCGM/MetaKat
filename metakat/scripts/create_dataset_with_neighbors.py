
neighbors = "/home/matko/Desktop/actual_neighbors/neighbors"
out = "/home/matko/Desktop/actual_neighbors/pages_with_existing_neighbors"

neigh_file = open(neighbors, "r")
out_file = open(out, "w")

main_id = None
has_prev = False
has_next = False

for i, line in enumerate(neigh_file):
    if line.startswith("main"):
        main_id = line.strip().replace("main: ", "")

    if line.startswith("prev"):
        if not "ERR" in line:
            has_prev = True

    if line.startswith("next"):
        if not "ERR" in line:
            has_next = True

    if (i + 1) % 3 == 0:
        if has_prev and has_next:
            out_file.write(main_id + "\n")

        main_id = None
        has_prev = False
        has_next = False


neigh_file.close()
out_file.close()