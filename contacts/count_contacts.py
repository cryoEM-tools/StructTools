import itertools
import mdtraj as md
import numpy as np
import pandas



convert_map_single_letter = {
    'ALA' : 'A', 'ASN' : 'N', 'CYS' : 'C', 'GLU' : 'E', 'HIS' : 'H',
    'LEU' : 'L', 'MET' : 'M', 'PRO' : 'P', 'THR' : 'T', 'TYR' : 'Y',
    'ARG' : 'R', 'ASP' : 'D', 'GLN' : 'Q', 'GLY' : 'G', 'ILE' : 'I',
    'LYS' : 'K', 'PHE' : 'F', 'SER' : 'S', 'TRP' : 'W', 'VAL' : 'V'}

convert_map_fancy = {
    'ALA' : 'Ala', 'ASN' : 'Asn', 'CYS' : 'Cys', 'GLU' : 'Glu', 'HIS' : 'His',
    'LEU' : 'Leu', 'MET' : 'Met', 'PRO' : 'Pro', 'THR' : 'Thr', 'TYR' : 'Tyr',
    'ARG' : 'Arg', 'ASP' : 'Asp', 'GLN' : 'Gln', 'GLY' : 'Gly', 'ILE' : 'Ile',
    'LYS' : 'Lys', 'PHE' : 'Phe', 'SER' : 'Ser', 'TRP' : 'Trp', 'VAL' : 'Val'}



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


def count_contacts_distance(struct0, struct1, cutoff=0.39, **kwargs):
    """Count contacts between struct0 and struct1.

    Inputs
    ----------
    struct0 : md.Trajectory,
        First structure for calculating contacts.
    struct1 : md.Trajectory,
        Second structure for calculating contacts.
    cutoff : float, default=0.39,
        The threshold distance between atoms to count a contact.

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
    
    # count contacts
    contacts_struct0, contacts_struct1 = _count_contacts_from_mask(
        struct0, struct1, contacts_mask, **kwargs)

    return contacts_struct0, contacts_struct1


def pairwise_VDW_radii(struct0, struct1):
    """Generates pairwise van der Waal radii between 2 sets of atoms.

    Inputs
    ----------
    struct0 : md.Trajectory,
        First structure for determining VDW radii.
    struct1 : md.Trajectory,
        Second structure for determining VDW radii.

    Returns
    ----------
    VDW_radii : nd.array, shape=(n_atoms0, n_atoms1,)
        pairwise sum of van der Waal radii between struct0 and struct1.
    """

    # obtain struct VDW radii
    VDW_radii_struct0 = np.array([a.element.radius for a in struct0.top.atoms])
    VDW_radii_struct1 = np.array([a.element.radius for a in struct1.top.atoms])

    # combine and sum radii
    VDW_expanded = np.array([[v]*VDW_radii_struct1.shape[0] for v in VDW_radii_struct0])
    VDW_radii = VDW_expanded + VDW_radii_struct1[None,:]

    return VDW_radii


def count_contacts_VDW(struct0, struct1, VDW_dist=1.4, atoms='all', **kwargs):
    """Count contacts between struct0 and struct1.

    Inputs
    ----------
    struct0 : md.Trajectory,
        First structure for calculating contacts.
    struct1 : md.Trajectory,
        Second structure for calculating contacts.
    VDW_frac_cutoff : float, default=1.4,
        The threshold van der Waals cutoff, expressed in fraction of
        van der Waals radii.
    atoms : str, choices=['all', 'heavy'],
        Optionally return residue-residue counts, or atom-atom counts.

    Returns
    ----------
    contacts_struct0 : nd.array, shape=(n_atoms_struct0,) or (n_residues_struct0,),
        Array detailing number of contacts for each atom or residue in struct0.
    contacts_struct1 : nd.array, shape=(n_atoms_struct1,) or (n_residues_struct1,),
        Array detailing number of contacts for each atom or residue in struct1.
    """

    assert atoms in ['all', 'heavy']

    if atoms == 'heavy':
        struct0 = struct0.atom_slice(struct0.top.select_atom_indices('heavy'))
        struct1 = struct1.atom_slice(struct1.top.select_atom_indices('heavy'))

    # get pairwise distance matrix
    pairwise_dist_mat = pairwise_dists(struct0, struct1)

    # get pairwise VDW radii
    VDW_radii = pairwise_VDW_radii(struct0, struct1)
    
    # obtain contacts
    contacts_mask = pairwise_dist_mat / VDW_radii <= VDW_dist

    # count contacts
    contacts_struct0, contacts_struct1 = _count_contacts_from_mask(
        struct0, struct1, contacts_mask, **kwargs)

    return contacts_struct0, contacts_struct1


def pairwise_contacts(
        struct0, struct1, criterion='VDW', VDW_dist=1.4, dist_cutoff=0.39,
        atoms='all', mode='residue'):
    """List out all pairwise contacts between struct0 and struct 1

    """
    
    # get pairwise distance matrix
    pairwise_dist_mat = pairwise_dists(struct0, struct1)

    if criterion == 'VDW':

        # get pairwise VDW radii
        VDW_radii = pairwise_VDW_radii(struct0, struct1)
    
        # obtain contacts
        contacts_mask = pairwise_dist_mat / VDW_radii <= VDW_dist
    
    elif criterion == 'dist':
    
        # obtain contacts
        contacts_mask = pairwise_dist_mat <= dist_cutoff

    if mode == 'atom':
        names0 = np.array([str(a) for a in struct0.top.atoms])
        names1 = np.array([str(a) for a in struct1.top.atoms])
        iis_contacts = np.where(contacts_mask > 0)
        n_contacts = contacts_mask[iis_contacts]

    elif mode == 'residue':
        names0 = np.array([str(r) for r in struct0.top.residues])
        names1 = np.array([str(r) for r in struct1.top.residues])

        resi_iis_struct0 = [[a.index for a in r.atoms] for r in struct0.top.residues]
        resi_iis_struct1 = [[a.index for a in r.atoms] for r in struct1.top.residues]

        n_resis0 = len(resi_iis_struct0)
        n_resis1 = len(resi_iis_struct1)
        contacts_mask_residues = np.zeros((n_resis0, n_resis1), dtype=int)
        for i in np.arange(n_resis0):
            for j in np.arange(n_resis1):
                x,y = np.array(
                    list(
                        itertools.product(
                            resi_iis_struct0[i],
                            resi_iis_struct1[j]))).T
                contacts_mask_residues[i,j] = contacts_mask[x,y].sum()
        iis_contacts = np.where(contacts_mask_residues > 0)
        n_contacts = contacts_mask_residues[iis_contacts]

    contact_pairs = np.vstack(
        [names0[iis_contacts[0]], names1[iis_contacts[1]]]).T

    return contact_pairs, n_contacts


def _count_contacts_from_mask(struct0, struct1, contacts_mask, mode='residue'):
    """Helper function to count contacts from a pairwise atomic mask

    Inputs
    ----------
    struct0 : md.Trajectory,
        First structure to use for topology.
    struct1 : md.Trajectory,
        Second structure to use for topology.
    contacts_mask : boolean-nd.array, shape=(n_atoms0, n_atoms1),
        Mask for counting contacts between struct0 and struct1.

    Returns
    ----------
    contacts_struct0 : nd.array, shape=(n_atoms_struct0,) or (n_residues_struct0,),
        Array detailing number of contacts for each atom or residue in struct0.
    contacts_struct1 : nd.array, shape=(n_atoms_struct1,) or (n_residues_struct1,),
        Array detailing number of contacts for each atom or residue in struct1.
    """
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



class Contacts():

    def __init__(self, struct0, struct1, name0='struct0', name1='struct1', **kwargs):
        self.struct0 = struct0
        self.struct1 = struct1

        self.name0 = name0
        self.name1 = name1

        self.pairs = None
        self.counts = None

        self.unique0 = None
        self.counts0 = None

        self.unique1 = None
        self.counts1 = None

    def compute_pairwise_contacts(self, **kwargs):
        self.pairs, self.counts = pairwise_contacts(self.struct0, self.struct1, **kwargs)

    def bincount_contacts(self, **kwargs):
        if self.pairs is None or self.counts is None:
            self.compute_pairwise_contacts(**kwargs)
        self.unique0 = np.unique(self.pairs[:,0])
        self.unique1 = np.unique(self.pairs[:,1])
        self.counts0 = np.array(
            [np.sum(self.pairs[:,0] == u) for u in self.unique0])
        self.counts1 = np.array(
            [np.sum(self.pairs[:,1] == u) for u in self.unique1])

    def __repr__(self):
        out = "Contacts(%s, %s)" % (self.name0, self.name1)
        return out

    def output(self, mode='table', residues='block', sort=True, **kwargs):

        if self.unique0 is None or self.counts0 is None:
            self.bincount_contacts(**kwargs)

        assert residues in ['block', 'fancy', 'single']

        resis0 = np.array([str(s[:3]) for s in self.unique0])
        splits0 = np.array([s[3:].split("-") for s in self.unique0])
        if splits0.shape[1] == 2:
            atoms0 = np.array([str(s[3:].split("-")[1]) for s in self.unique0])
            resSeqs0 = np.array([int(s[3:].split("-")[0]) for s in self.unique0])
        else:
            atoms0 = None
            resSeqs0 = np.array([int(s[3:]) for s in self.unique0])

        resis1 = np.array([str(s[:3]) for s in self.unique1])
        splits1 = np.array([s[3:].split("-") for s in self.unique1])
        if splits1.shape[1] == 2:
            atoms1 = np.array([str(s[3:].split("-")[1]) for s in self.unique1])
            resSeqs1 = np.array([int(s[3:].split("-")[0]) for s in self.unique1])
        else:
            atoms1 = None
            resSeqs1 = np.array([int(s[3:]) for s in self.unique1])

        if sort:
            iis0 = np.argsort(resSeqs0)
            iis1 = np.argsort(resSeqs1)
        else:
            iis0 = np.arange(resSeqs0.shape[0])
            iis1 = np.arange(resSeqs1.shape[0])

        if residues == 'fancy':
            resis0 = np.array([convert_map_fancy[l] for l in resis0])
            resis1 = np.array([convert_map_fancy[l] for l in resis1])
        elif residues == 'single':
            resis0 = np.array([convert_map_single[l] for l in resis0])
            resis1 = np.array([convert_map_single[l] for l in resis1])
            
        if mode == 'table':
            if atoms0 is not None:
                df_resis0 = np.array(
                    ['%s%s-%s' % (aa,num,atom) for aa,num,atom in zip(resis0, resSeqs0, atoms0)])
            else:
                df_resis0 = np.array(
                    ['%s%s' % (aa,num) for aa,num in zip(resis0, resSeqs0)])
            df0 = pandas.DataFrame(
                {
                    'residues' : df_resis0[iis0],
                    'counts' : self.counts0[iis0]
                })

            if atoms1 is not None:
                df_resis1 = np.array(
                    ['%s%s-%s' % (aa,num,atom) for aa,num,atom in zip(resis1, resSeqs1, atoms1)])
            else:
                df_resis1 = np.array(
                    ['%s%s' % (aa,num) for aa,num in zip(resis1, resSeqs1)])
            df1 = pandas.DataFrame(
                {
                    'residues' : df_resis1[iis1],
                    'counts' : self.counts1[iis1]
                })

            print(
                "".join(
                    [
                        "%s\n" % self.name0,
                        repr(df0),
                        "\n\n%s\n" % self.name1,
                        repr(df1)]))


        elif mode == 'fancy':
            if atoms0 is not None:
                fancy_data0 = zip(resis0[iis0], resSeqs0[iis0], atoms0, self.counts0[iis0])
                fancy_out0 = ", ".join(
                    [
                        '%s%s-%s(%d)' % (aa,num,atom,contacts)
                        for aa,num,atom,contacts in fancy_data0])
            else:
                fancy_data0 = zip(resis0[iis0], resSeqs0[iis0], self.counts0[iis0])
                fancy_out0 = ", ".join(
                    [
                        '%s%s(%d)' % (aa,num,contacts)
                        for aa,num,contacts in fancy_data0])

            if atoms1 is not None:
                fancy_data1 = zip(resis1[iis1], resSeqs1[iis1], atoms1, self.counts1[iis1])
                fancy_out1 = ", ".join(
                    [
                        '%s%s-%s(%d)' % (aa,num,atom,contacts)
                        for aa,num,atom,contacts in fancy_data1])
            else:
                fancy_data1 = zip(resis1[iis1], resSeqs1[iis1], self.counts1[iis1])
                fancy_out1 = ", ".join(
                    [
                        '%s%s(%d)' % (aa,num,contacts)
                        for aa,num,contacts in fancy_data1])
            print(
                "".join(
                    [
                        "%s\n" % self.name0,
                        fancy_out0,
                        "\n\n%s\n" % self.name1,
                        fancy_out1]))

        elif mode == 'chimerax':
            print(resis0, resSeqs0, resis1, resSeqs1)
            chimerax_out = "".join(
                [
                    "%s\n" % self.name0,
                    ", ".join(np.array(resSeqs0[iis0],dtype=str)),
                    "\n\n%s\n" % self.name1,
                    ", ".join(np.array(resSeqs1[iis1], dtype=str))])
            print(chimerax_out)
                
        elif mode == 'pymol':
            pymol_out = "".join(
                [
                    "%s\n" % self.name0,
                    "+".join(np.array(resSeqs0[iis0],dtype=str)),
                    "\n\n%s\n" % self.name1,
                    "+".join(np.array(resSeqs1[iis1],dtype=str))])
            print(pymol_out)
