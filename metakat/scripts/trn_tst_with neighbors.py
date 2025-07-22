pages = "/home/matko/Desktop/pages_with_neighbors"

trn = "/home/matko/Desktop/bakalarka/data/ann/pages.trn"
tst = "/home/matko/Desktop/bakalarka/data/ann/pages.tst"

neighbors_trn = "/home/matko/Desktop/trn.neighbors"
neighbors_tst = "/home/matko/Desktop/tst.neighbors"

trn_file = open(trn, "r")
tst_file = open(tst, "r")
trn_out = open(neighbors_trn, "w")
tst_out = open(neighbors_tst, "w")

for i, line in enumerate(open(pages, "r")):
    all_id = line.strip()

    trn_file.seek(0)
    for trn_line in trn_file:
        trn_id = trn_line.strip().split()[0]

        if all_id == trn_id:
            trn_out.write(trn_line)

    tst_file.seek(0)
    for tst_line in tst_file:
        tst_id = tst_line.strip().split()[0]

        if all_id == tst_id:
            tst_out.write(tst_line)

trn_file.close()
tst_file.close()
trn_out.close()
tst_out.close()