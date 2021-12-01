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


small_number = 1E-9
def _calc_sasa_apo_holo(struct0, struct1, mode='residue', **kwargs):
    """Calculates the solvent exposure of two structures and their differences when bound.

    Inputs
    ----------
    struct0 : md.Trajectory,
        First structure for calculating SASAs.
    struct1 : md.Trajectory,
        Second structure for calculating SASAs.
    mode : str, choices=['atom', 'residue']
        Optionally return SASA by atom or residue.

    Returns
    ----------
    sasa_struct0_apo : nd.array, shape=(n_atoms,) or (n_residues),
        The SASA of struct0 in isolation, either by atom or residue.
    sasa_struct1_apo : nd.array, shape=(n_atoms,) or (n_residues),
        The SASA of struct1 in isolation, either by atom or residue.
    sasa_struct0_holo : nd.array, shape=(n_atoms,) or (n_residues),
        The SASA of struct0 in isolation, either by atom or residue.
    sasa_struct1_holo : nd.array, shape=(n_atoms,) or (n_residues),
        The SASA of struct1 in isolation, either by atom or residue.
    """ 
    # obtain combined structure (add struct1 to struct0)
    combined_struct_topol = struct0.topology.copy().join(struct1.topology)
    combined_struct_xyz = np.concatenate([struct0.xyz, struct1.xyz], axis=1)
    combined_struct = md.Trajectory(combined_struct_xyz, combined_struct_topol)
    
    # only residue mode is currently supported
    if mode == 'residue':
        
        # calculate SASAs
        sasa_struct0_apo = md.shrake_rupley(
            struct0, mode='residue', **kwargs)[0] + small_number
        sasa_struct1_apo = md.shrake_rupley(
            struct1, mode='residue', **kwargs)[0] + small_number
        sasa_combined_states = md.shrake_rupley(
            combined_struct, mode='residue', **kwargs)[0] + small_number
        
        # extract struct0 and struct1 from combined structure
        sasa_struct0_holo = sasa_combined_states[:struct0.n_residues]
        sasa_struct1_holo = sasa_combined_states[struct0.n_residues:]
    else:
        raise
        
    # return sasas        
    return sasa_struct0_apo, sasa_struct0_holo, sasa_struct1_apo, sasa_struct1_holo


def sasa_change_fraction(struct0, struct1, **kwargs):
    """Calculate the percentage SASA change from apo to holo.

    Inputs
    ----------
    struct0 : md.Trajectory,
        First structure for calculating SASAs.
    struct1 : md.Trajectory,
        Second structure for calculating SASAs.

    Returns
    ----------
    sasa_change_frac_struct0 : nd.array, shape=(n_atoms,) or (n_residues),
        The fraction change of SASA on struct0 apo v holo.
    sasa_change_frac_struct1 : nd.array, shape=(n_atoms,) or (n_residues),
        The fraction change of SASA on struct1 apo v holo.
    """

    # calculate SASAs
    sasa_struct0_apo, sasa_struct0_holo, sasa_struct1_apo, sasa_struct1_holo = \
        calc_sasa_apo_holo(struct0, struct1, **kwargs)

    # calculate fractional change
    sasa_change_struct0 = sasa_struct0_apo - sasa_struct0_holo
    sasa_change_struct1 = sasa_struct1_apo - sasa_struct1_holo
    sasa_change_frac_struct0 = (sasa_change_struct0 / sasa_struct0_apo)
    sasa_change_frac_struct1 = (sasa_change_struct1 / sasa_struct1_apo)
    return sasa_change_frac_struct0, sasa_change_frac_struct1


def bin_sasa_change(sasas):
    """Bins SASA values by PDBePISA convention,
    i.e. 10-20% -> 1, 20-30% -> 2, etc."""
    sasas *= 10
    sasas = np.array([int(sasa) for sasa in sasas])
    return sasas


def count_PISA_contacts(struct0, struct1, **kwargs):
    """Determine strength of residue-residue contacts using PISA conventions.
    Strength is reported in fraction of SASA change between apo-holo structures.

    Inputs
    ----------
    struct0 : md.Trajectory,
        First structure for calculating SASAs.
    struct1 : md.Trajectory,
        Second structure for calculating SASAs.

    Returns
    ----------
    PISA_SASA_struct0 : nd.array, shape=(n_residues,),
        Categorized contacts per residue for struct0.
    PISA_SASA_struct1 : nd.array, shape=(n_residues,),
        Categorized contacts per residue for struct1.
    """
    sasa_frac_struct0, sasa_frac_struct1 = sasa_change_fraction(
        struct0, struct1, **kwargs)
    PISA_SASA_struct0 = bin_sasa_change(sasa_frac_struct0)
    PISA_SASA_struct1 = bin_sasa_change(sasa_frac_struct1)
    return PISA_SASA_struct0, PISA_SASA_struct1
