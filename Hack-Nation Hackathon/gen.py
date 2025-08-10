import random
amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
structure_labels = ["H", "E", "C"] 

num_sequences = 100
min_len, max_len = 50, 120

with open("seq.txt", "w") as seq_file, open("label.txt", "w") as label_file:
    for _ in range(num_sequences):
        length = random.randint(min_len, max_len)
        seq = "".join(random.choices(amino_acids, k=length))
        labels = "".join(random.choices(structure_labels, k=length))
        seq_file.write(seq + "\n")
        label_file.write(labels + "\n")
