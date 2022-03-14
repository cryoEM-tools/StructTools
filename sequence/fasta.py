import numpy as np

def write_fasta(filename, seq, seq_title, n_resis=70, append=True):
    remaining_seq = copy.copy(seq)
    if append:
        with open(filename, "a") as f:
            f.write(header)
            while len(remaining_seq) > 0:
                f.write(remaining_seq[:n_resis] + "\n")
                remaining_seq = remaining_seq[n_resis:]
            f.write("\n")
    else:
        with open(filename, "w") as f:
            f.write(header)
            while len(remaining_seq) > 0:
                f.write(remaining_seq[:n_resis] + "\n")
                remaining_seq = remaining_seq[n_resis:]
            f.write("\n")
    return
