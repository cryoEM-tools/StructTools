import copy
import mdtraj as md
import numpy as np


def _join_tops(top1, top2, new_chain=True):
    if new_chain:
        new_top = top1.join(top2)
    else:
        new_top = copy.deepcopy(top1)
        chain = list(new_top.chains)[0]
        for r in top2.residues:
            new_res = new_top.add_residue(r.name, chain, resSeq=r.resSeq)
            for a in r.atoms:
                new_atom = new_top.add_atom(a.name, a.element, new_res, a.serial)
    return new_top


def join_tops(tops, new_chain=True):
    new_top = copy.deepcopy(tops[0])
    for n in np.arange(1, len(tops)):
        new_top = _join_tops(new_top, tops[n], new_chain=new_chain)
    return new_top


def join_structs(structs, new_chains=True):
    new_xyz = np.vstack([s.xyz[0] for s in structs])
    topols = [s.top for s in structs]
    new_tops = join_tops(topols, new_chain=new_chains)
    return md.Trajectory(new_xyz, topology=new_tops)
