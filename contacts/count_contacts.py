import numpy as np
import mdtraj as md




def pairwise_dists(struct0, struct1):
    """calculate pairwise distances between atoms in two structures.

    Inputs
    ----------
    struct0 : md.Trajectory
        First structure for calculating pairwise distances.
    struct1 : md.Trajectory

    Returns
    ----------
    dists : nd.array, shape=(n_states_struct0, n_states_struct1),
        Pairwise distances between struct 0 and 1. i,j corresponds to distance
        between ith atom in struct0 and jth atom in struct1.
    """
    diffs = np.abs(struct0.xyz[0][:,None,:] - struct1.xyz[0])
    dists = np.sqrt(np.einsum('ijk,ijk->ij', diffs, diffs))
    return dists


def min_dists(struct0, struct1, mode='residue'):
    """Obtain the minimum distance between residues or atoms in two structures.

    Inputs
    ----------
    struct0 : md.Trajectory,
        First structure for calculating minimum distances.
    struct1 : md.Trajectory,
        Second structure for calculating minimum distances.
    mode : str, options=['residue', 'atom'], default='residue,
        Calculate minimum distances either on a residue level, or
        on an atomic level.

    Returns
    ----------
    min_dists_struct0, nd.array, shape=(n_atoms_struct0,) or (n_residues_struct0,),
        The minimum distance between residue i in struct0 and any residue in struct1.
    min_dists_struct1, nd.array, shape=(n_atoms_struct1,) or (n_residues_struct1,),
        The minimum distance between residue i in struct1 and any residue in struct0.
    """
    # get pairwise distance matrix
    pairwise_dist_mat = pairwise_dists(struct0, struct1)
    
    # obtain minimum distances
    min_atomic_dists_struct0 = np.min(pairwise_dist_mat, axis=1)
    min_atomic_dists_struct1 = np.min(pairwise_dist_mat, axis=0)

    # if mode is residues, return minimum residue distance
    if mode == 'residue':
        resSeqs_struct0 = [r.resSeq for r in struct0.top.residues]
        resSeqs_struct1 = [r.resSeq for r in struct1.top.residues]
        resi_iis_struct0 = [[a.index for a in r.atoms] for r in struct0.top.residues]
        resi_iis_struct1 = [[a.index for a in r.atoms] for r in struct1.top.residues]
        min_dists_struct0 = np.array(
            [
                min_atomic_dists_struct0[iis].min() for iis in resi_iis_struct0])
        min_dists_struct1 = np.array(
            [
                min_atomic_dists_struct1[iis].min() for iis in resi_iis_struct1])
    else:
        min_dists_struct0 = min_atomic_dists_struct0
        min_dists_struct1 = min_atomic_dists_struct1
    return min_dists_struct0, min_dists_struct1


def count_contacts_distance(struct0, struct1, cutoff=0.39, mode='residue'):
    """Count contacts between struct0 and struct1.

    Inputs
    ----------
    struct0 : md.Trajectory,
        First structure for calculating contacts.
    struct1 : md.Trajectory,
        Second structure for calculating contacts.
    cutoff : float, default=0.39,
        The threshold distance between atoms to count a contact.
    mode : str, choices=['atom', 'residue'],
        Optionally return residue-residue counts, or atom-atom counts.

    Returns
    ----------
    contacts_struct0 : nd.array, shape=(n_atoms_struct0,) or (n_residues_struct0,),
        Array detailing number of contacts for each atom or residue in struct0.
    contacts_struct1 : nd.array, shape=(n_atoms_struct1,) or (n_residues_struct1,),
        Array detailing number of contacts for each atom or residue in struct1.
    """
    # get pairwise distance matrix
    pairwise_dist_mat = pairwise_dists(struct0, struct1)
    
    # obtain contacts
    contacts_mask = pairwise_dist_mat <= cutoff
    atomic_contacts_struct0 = np.sum(contacts_mask, axis=1)
    atomic_contacts_struct1 = np.sum(contacts_mask, axis=0)

    # if mode is residues, return minimum residue distance
    if mode == 'residue':
        resSeqs_struct0 = [r.resSeq for r in struct0.top.residues]
        resSeqs_struct1 = [r.resSeq for r in struct1.top.residues]
        resi_iis_struct0 = [[a.index for a in r.atoms] for r in struct0.top.residues]
        resi_iis_struct1 = [[a.index for a in r.atoms] for r in struct1.top.residues]
        contacts_struct0 = np.array(
            [
                atomic_contacts_struct0[iis].sum() for iis in resi_iis_struct0])
        contacts_struct1 = np.array(
            [
                atomic_contacts_struct1[iis].sum() for iis in resi_iis_struct1])
    else:
        contacts_struct0 = atomic_contacts_struct0
        contacts_struct1 = atomic_contacts_struct1
    return contacts_struct0, contacts_struct1
