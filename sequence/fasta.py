import copy
import numpy as np
import os

def _write_fasta(filename, seq, seq_title, n_resis=70, append=True):
    remaining_seq = copy.copy(seq)
    header = '>%s\n' % seq_title
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

def write_fasta(filename, seqs, seq_titles, n_resis=70, overwrite=False):

    if not overwrite:
        assert not os.path.exists(filename)

    assert len(seqs) == len(seq_titles)

    for n in np.arange(len(seqs)):
        if n == 0:
            _write_fasta(
                filename, seqs[n], seq_titles[n], n_resis=n_resis, append=False)
        else:
            _write_fasta(
                filename, seqs[n], seq_titles[n], n_resis=n_resis, append=True)
    return

