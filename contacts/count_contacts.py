import itertools
import mdtraj as md
import numpy as np
import pandas
from enspara import ra


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


class color:
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'

def pairwise_dists(structA, structB):
    """calculate pairwise distances between atoms in two structures.

    Inputs
    ----------
    structA : md.Trajectory,
        First structure for calculating pairwise distances.
    structB : md.Trajectory,

    Returns
    ----------
    dists : nd.array, shape=(n_frames, n_states_structA, n_states_structB),
        Pairwise distances between struct 0 and 1. i,j corresponds to distance
        between ith atom in structA and jth atom in structB.
    """
    assert structA.xyz.shape[0] == structB.xyz.shape[0]
    diffs = np.abs(
        [xyzA[:,None,:] - xyzB for xyzA,xyzB in zip(structA.xyz, structB.xyz)])
    dists = np.sqrt(np.einsum('ijkl,ijkl->ijk', diffs, diffs))
    return dists


def min_dists(structA, structB, mode='residue'):
    """Obtain the minimum distance between residues or atoms in two structures.

    Inputs
    ----------
    structA : md.Trajectory,
        First structure for calculating minimum distances.
    structB : md.Trajectory,
        Second structure for calculating minimum distances.
    mode : str, options=['residue', 'atom'], default='residue,
        Calculate minimum distances either on a residue level, or
        on an atomic level.

    Returns
    ----------
    min_dists_structA, nd.array, shape=(n_atoms_structA,) or (n_residues_structA,),
        The minimum distance between residue i in structA and any residue in structB.
    min_dists_structB, nd.array, shape=(n_atoms_structB,) or (n_residues_structB,),
        The minimum distance between residue i in structB and any residue in structA.
    """
    # get pairwise distance matrix
    pairwise_dist_mat = pairwise_dists(structA, structB)
    
    # obtain minimum distances
    min_atomic_dists_structA = np.min(pairwise_dist_mat, axis=1)
    min_atomic_dists_structB = np.min(pairwise_dist_mat, axis=0)

    # if mode is residues, return minimum residue distance
    if mode == 'residue':
        resSeqs_structA = [r.resSeq for r in structA.top.residues]
        resSeqs_structB = [r.resSeq for r in structB.top.residues]
        resi_iis_structA = [[a.index for a in r.atoms] for r in structA.top.residues]
        resi_iis_structB = [[a.index for a in r.atoms] for r in structB.top.residues]
        min_dists_structA = np.array(
            [
                min_atomic_dists_structA[iis].min() for iis in resi_iis_structA])
        min_dists_structB = np.array(
            [
                min_atomic_dists_structB[iis].min() for iis in resi_iis_structB])
    else:
        min_dists_structA = min_atomic_dists_structA
        min_dists_structB = min_atomic_dists_structB
    return min_dists_structA, min_dists_structB


def count_contacts_distance(structA, structB, cutoff=0.39, **kwargs):
    """Count contacts between structA and structB.

    Inputs
    ----------
    structA : md.Trajectory,
        First structure for calculating contacts.
    structB : md.Trajectory,
        Second structure for calculating contacts.
    cutoff : float, default=0.39,
        The threshold distance between atoms to count a contact.

    Returns
    ----------
    contacts_structA : nd.array, shape=(n_atoms_structA,) or (n_residues_structA,),
        Array detailing number of contacts for each atom or residue in structA.
    contacts_structB : nd.array, shape=(n_atoms_structB,) or (n_residues_structB,),
        Array detailing number of contacts for each atom or residue in structB.
    """
    # get pairwise distance matrix
    pairwise_dist_mat = pairwise_dists(structA, structB)
    
    # obtain contacts
    contacts_mask = pairwise_dist_mat <= cutoff
    
    # count contacts
    contacts_structA, contacts_structB = _count_contacts_from_mask(
        structA, structB, contacts_mask, **kwargs)

    return contacts_structA, contacts_structB


def pairwise_VDW_radii(structA, structB):
    """Generates pairwise van der Waal radii between 2 sets of atoms.

    Inputs
    ----------
    structA : md.Trajectory,
        First structure for determining VDW radii.
    structB : md.Trajectory,
        Second structure for determining VDW radii.

    Returns
    ----------
    VDW_radii : nd.array, shape=(n_atomsA, n_atomsB,)
        pairwise sum of van der Waal radii between structA and structB.
    """

    # obtain struct VDW radii
    VDW_radii_structA = np.array([a.element.radius for a in structA.top.atoms])
    VDW_radii_structB = np.array([a.element.radius for a in structB.top.atoms])

    # combine and sum radii
    VDW_expanded = np.array([[v]*VDW_radii_structB.shape[0] for v in VDW_radii_structA])
    VDW_radii = VDW_expanded + VDW_radii_structB[None,:]

    return VDW_radii


def count_contacts_VDW(structA, structB, VDW_dist=1.4, atoms='all', **kwargs):
    """Count contacts between structA and structB.

    Inputs
    ----------
    structA : md.Trajectory,
        First structure for calculating contacts.
    structB : md.Trajectory,
        Second structure for calculating contacts.
    VDW_frac_cutoff : float, default=1.4,
        The threshold van der Waals cutoff, expressed in fraction of
        van der Waals radii.
    atoms : str, choices=['all', 'heavy'],
        Optionally return residue-residue counts, or atom-atom counts.

    Returns
    ----------
    contacts_structA : nd.array, shape=(n_atoms_structA,) or (n_residues_structA,),
        Array detailing number of contacts for each atom or residue in structA.
    contacts_structB : nd.array, shape=(n_atoms_structB,) or (n_residues_structB,),
        Array detailing number of contacts for each atom or residue in structB.
    """

    assert atoms in ['all', 'heavy']

    if atoms == 'heavy':
        structA = structA.atom_slice(structA.top.select_atom_indices('heavy'))
        structB = structB.atom_slice(structB.top.select_atom_indices('heavy'))

    # get pairwise distance matrix
    pairwise_dist_mat = pairwise_dists(structA, structB)

    # get pairwise VDW radii
    VDW_radii = pairwise_VDW_radii(structA, structB)
    
    # obtain contacts
    contacts_mask = pairwise_dist_mat / VDW_radii <= VDW_dist

    # count contacts
    contacts_structA, contacts_structB = _count_contacts_from_mask(
        structA, structB, contacts_mask, **kwargs)

    return contacts_structA, contacts_structB


def pairwise_contacts(
        structA, structB, criterion='VDW', VDW_dist=1.4, dist_cutoff=0.39,
        atoms='all', mode='residue'):
    """List out all pairwise contacts between structA and struct 1

    Inputs
    ----------
    structA : md.Trajectory,
        The first structure to use for calculating close contacts.
    structB : md.Trajectory,
        The second structure to use for calculating close contacts.
    criterion : str, choices=['VDW', 'dist'], default='VDW',
        Criterion for determining close contacts, either as a fraction
        of the van der Waals radii, or within a particular distance.
    VDW_dist : float, default=1.4,
        The fraction of van der Waals radii to use as a cutoff for
        determining close-contacts. i.e. considered a contact if a
        distance pair is less than 1.4x the sum of the van der Waals
        radii (default).
    dist_cutoff : float, default=0.39,
        The distance cutoff for determining a close-contact. Default is
        set to 0.39 nm.
    atoms : str, choices=['all', 'heavy', 'backbone', 'Ca'], default='all',
        Atoms to include for close contacts, 'all', or 'heavy'. Use all
        atoms or only heavy atoms.
    mode : str, choices=['residue', 'atom'], default='residue',
        Option for computing granularity of contacts. Can count based
        on residues or per atom pair.
    """

    # select atom indices
    if atoms == 'heavy':
        structA = structA.atom_slice(structA.top.select_atom_indices('heavy'))
        structB = structB.atom_slice(structB.top.select_atom_indices('heavy'))
    elif atoms == 'backbone':
        structA = structA.atom_slice(structA.top.select_atom_indices('minimal'))
        structB = structB.atom_slice(structB.top.select_atom_indices('minimal'))
    elif atoms == 'Ca':
        structA = structA.atom_slice(structA.top.select('name CA'))
        structB = structB.atom_slice(structB.top.select('name CA'))
    elif atoms == 'all':
        pass
    else:
        raise
        
    
    # get pairwise distance matrix
    pairwise_dist_mat = pairwise_dists(structA, structB)

    if criterion == 'VDW':

        # get pairwise VDW radii
        VDW_radii = pairwise_VDW_radii(structA, structB)
    
        # obtain contacts
        contacts_mask = pairwise_dist_mat / VDW_radii <= VDW_dist
    
    elif criterion == 'dist':
    
        # obtain contacts
        contacts_mask = pairwise_dist_mat <= dist_cutoff

    if mode == 'atom':
        names0 = np.array([str(a) for a in structA.top.atoms])
        names1 = np.array([str(a) for a in structB.top.atoms])
        iis_contacts = np.where(contacts_mask > 0)
        n_contacts = contacts_mask[iis_contacts]

    elif mode == 'residue':
        names0 = np.array([str(r) for r in structA.top.residues])
        names1 = np.array([str(r) for r in structB.top.residues])

        resi_iis_structA = [[a.index for a in r.atoms] for r in structA.top.residues]
        resi_iis_structB = [[a.index for a in r.atoms] for r in structB.top.residues]

        n_resisA = len(resi_iis_structA)
        n_resisB = len(resi_iis_structB)
        contacts_mask_residues = np.zeros(
            (structA.n_frames, n_resisA, n_resisB), dtype=int)
        for i in np.arange(n_resisA):
            for j in np.arange(n_resisB):
                x,y = np.array(
                    list(
                        itertools.product(
                            resi_iis_structA[i],
                            resi_iis_structB[j]))).T
                contacts_mask_residues[:,i,j] = contacts_mask[:,x,y].sum(axis=1)
        iis_contacts = np.where(contacts_mask_residues > 0)
        n_contacts = contacts_mask_residues[iis_contacts]

    if n_contacts.shape[0] > 0:
        lengths = np.bincount(iis_contacts[0], minlength=pairwise_dist_mat.shape[0])
        n_contacts = ra.RaggedArray(n_contacts, lengths=lengths)

        named_pairs = np.vstack(
            [names0[iis_contacts[1]], names1[iis_contacts[2]]]).T
        contact_pairs = ra.RaggedArray(named_pairs, lengths=lengths)
    else:
        contact_pairs = None
        n_contacts = None

    return contact_pairs, n_contacts


def _count_contacts_from_mask(structA, structB, contacts_mask, mode='residue'):
    """Helper function to count contacts from a pairwise atomic mask

    Inputs
    ----------
    structA : md.Trajectory,
        First structure to use for topology.
    structB : md.Trajectory,
        Second structure to use for topology.
    contacts_mask : boolean-nd.array, shape=(n_atomsA, n_atomsB),
        Mask for counting contacts between structA and structB.

    Returns
    ----------
    contacts_structA : nd.array, shape=(n_atoms_structA,) or (n_residues_structA,),
        Array detailing number of contacts for each atom or residue in structA.
    contacts_structB : nd.array, shape=(n_atoms_structB,) or (n_residues_structB,),
        Array detailing number of contacts for each atom or residue in structB.
    """
    atomic_contacts_structA = np.sum(contacts_mask, axis=1)
    atomic_contacts_structB = np.sum(contacts_mask, axis=0)

    # if mode is residues, return minimum residue distance
    if mode == 'residue':
        resSeqs_structA = [r.resSeq for r in structA.top.residues]
        resSeqs_structB = [r.resSeq for r in structB.top.residues]
        resi_iis_structA = [[a.index for a in r.atoms] for r in structA.top.residues]
        resi_iis_structB = [[a.index for a in r.atoms] for r in structB.top.residues]
        contacts_structA = np.array(
            [
                atomic_contacts_structA[iis].sum() for iis in resi_iis_structA])
        contacts_structB = np.array(
            [
                atomic_contacts_structB[iis].sum() for iis in resi_iis_structB])
    else:
        contacts_structA = atomic_contacts_structA
        contacts_structB = atomic_contacts_structB
    return contacts_structA, contacts_structB


small_number = 1E-9
def _calc_sasa_apo_holo(structA, structB, mode='residue', **kwargs):
    """Calculates the solvent exposure of two structures and their differences when bound.

    Inputs
    ----------
    structA : md.Trajectory,
        First structure for calculating SASAs.
    structB : md.Trajectory,
        Second structure for calculating SASAs.
    mode : str, choices=['atom', 'residue']
        Optionally return SASA by atom or residue.

    Returns
    ----------
    sasa_structA_apo : nd.array, shape=(n_atoms,) or (n_residues),
        The SASA of structA in isolation, either by atom or residue.
    sasa_structB_apo : nd.array, shape=(n_atoms,) or (n_residues),
        The SASA of structB in isolation, either by atom or residue.
    sasa_structA_holo : nd.array, shape=(n_atoms,) or (n_residues),
        The SASA of structA in isolation, either by atom or residue.
    sasa_structB_holo : nd.array, shape=(n_atoms,) or (n_residues),
        The SASA of structB in isolation, either by atom or residue.
    """ 
    # obtain combined structure (add structB to structA)
    combined_struct_topol = structA.topology.copy().join(structB.topology)
    combined_struct_xyz = np.concatenate([structA.xyz, structB.xyz], axis=1)
    combined_struct = md.Trajectory(combined_struct_xyz, combined_struct_topol)
    
    # only residue mode is currently supported
    if mode == 'residue':
        
        # calculate SASAs
        sasa_structA_apo = md.shrake_rupley(
            structA, mode='residue', **kwargs)[0] + small_number
        sasa_structB_apo = md.shrake_rupley(
            structB, mode='residue', **kwargs)[0] + small_number
        sasa_combined_states = md.shrake_rupley(
            combined_struct, mode='residue', **kwargs)[0] + small_number
        
        # extract structA and structB from combined structure
        sasa_structA_holo = sasa_combined_states[:structA.n_residues]
        sasa_structB_holo = sasa_combined_states[structA.n_residues:]
    else:
        raise
        
    # return sasas        
    return sasa_structA_apo, sasa_structA_holo, sasa_structB_apo, sasa_structB_holo


def sasa_change_fraction(structA, structB, **kwargs):
    """Calculate the percentage SASA change from apo to holo.

    Inputs
    ----------
    structA : md.Trajectory,
        First structure for calculating SASAs.
    structB : md.Trajectory,
        Second structure for calculating SASAs.

    Returns
    ----------
    sasa_change_frac_structA : nd.array, shape=(n_atoms,) or (n_residues),
        The fraction change of SASA on structA apo v holo.
    sasa_change_frac_structB : nd.array, shape=(n_atoms,) or (n_residues),
        The fraction change of SASA on structB apo v holo.
    """

    # calculate SASAs
    sasa_structA_apo, sasa_structA_holo, sasa_structB_apo, sasa_structB_holo = \
        calc_sasa_apo_holo(structA, structB, **kwargs)

    # calculate fractional change
    sasa_change_structA = sasa_structA_apo - sasa_structA_holo
    sasa_change_structB = sasa_structB_apo - sasa_structB_holo
    sasa_change_frac_structA = (sasa_change_structA / sasa_structA_apo)
    sasa_change_frac_structB = (sasa_change_structB / sasa_structB_apo)
    return sasa_change_frac_structA, sasa_change_frac_structB


def bin_sasa_change(sasas):
    """Bins SASA values by PDBePISA convention,
    i.e. 10-20% -> 1, 20-30% -> 2, etc."""
    sasas *= 10
    sasas = np.array([int(sasa) for sasa in sasas])
    return sasas


def count_PISA_contacts(structA, structB, **kwargs):
    """Determine strength of residue-residue contacts using PISA conventions.
    Strength is reported in fraction of SASA change between apo-holo structures.

    Inputs
    ----------
    structA : md.Trajectory,
        First structure for calculating SASAs.
    structB : md.Trajectory,
        Second structure for calculating SASAs.

    Returns
    ----------
    PISA_SASA_structA : nd.array, shape=(n_residues,),
        Categorized contacts per residue for structA.
    PISA_SASA_structB : nd.array, shape=(n_residues,),
        Categorized contacts per residue for structB.
    """
    sasa_frac_structA, sasa_frac_structB = sasa_change_fraction(
        structA, structB, **kwargs)
    PISA_SASA_structA = bin_sasa_change(sasa_frac_structA)
    PISA_SASA_structB = bin_sasa_change(sasa_frac_structB)
    return PISA_SASA_structA, PISA_SASA_structB


def _table_df(resis, resSeqs, atoms, counts):
    if atoms is not None:
        df_resis = np.array(
            ['%s%s-%s' % (aa,num,atom) for aa,num,atom in zip(resis, resSeqs, atoms)])
    else:
        df_resis = np.array(
            ['%s%s' % (aa,num) for aa,num in zip(resis, resSeqs)])
    df = pandas.DataFrame(
        {
            'residues' : df_resis,
            'counts' : counts
        })
    return df


def _fancy_df(resis, resSeqs, atoms, counts):
    if atoms is not None:
        fancy_data = zip(resis, resSeqs, atoms, counts)
        fancy_out = ", ".join(
            [
                '%s%s-%s(%d)' % (aa,num,atom,contacts)
                for aa,num,atom,contacts in fancy_data])
    else:
        fancy_data = zip(resis, resSeqs, counts)
        fancy_out = ", ".join(
            [
                '%s%s(%d)' % (aa,num,contacts)
                for aa,num,contacts in fancy_data])
    return fancy_out


def _parse_residues(unique_resis):
    resis = np.array([str(s[:3]) for s in unique_resis])
    splits = np.array([s[3:].split("-") for s in unique_resis])
    if splits.shape[1] == 2:
        atoms = np.array([str(s[3:].split("-")[1]) for s in unique_resis])
        resSeqs = np.array([int(s[3:].split("-")[0]) for s in unique_resis])
    else:
        atoms = None
        resSeqs = np.array([int(s[3:]) for s in unique_resis])
    return resis, resSeqs, atoms


def _add_spacing(data, to_add=''):
    split_data = data.split("\n")
    spaced_split_data = ['%s%s' % (to_add,dat) for dat in split_data]
    return "\n".join(spaced_split_data)


class Contacts():
    """Class for computing and analyzing protein-protein contacts.

    Attributes
    ----------
    structA : md.Trajectory,
        First structure for computing contacts.
    structB : md.Trajectory,
        Second structure for computing contacts.
    nameA : str, default='structA',
        Name of the first structure for computing contacts.
    nameB : str, default='structB',
        Name of the second structure for computing contacts.
    pairs : nd.array, shape=(n_contact_pairs, 2),
        List of named contact pairs.
    counts : nd.array, shape=(n_contact_pairs, 2),
        Number of observed contacts per pair.
    uniqueA : nd.array, shape=(n_contact_pairs, ),
        Unique names of contacts on structure 1.
    countsA : nd.array, shape=(n_contact_pairs, ),
        Number of contacts for each pair on structure 1.
    uniqueB : nd.array, shape=(n_contact_pairs, ),
        Unique names of contacts on structure 2.
    countsB : nd.array, shape=(n_contact_pairs, ),
        Number of contacts for each pair on structure 2.
    """
    def __init__(self, structA, structB, nameA='structA', nameB='structB', **kwargs):
        self.structA = structA
        self.structB = structB

        assert self.structA.n_frames == self.structB.n_frames
        self.n_confs = self.structA.n_frames

        self.nameA = nameA
        self.nameB = nameB

        self.pairs = None
        self.counts = None

        self.uniqueA = None
        self.countsA = None
        self.resSeqsA = None

        self.uniqueB = None
        self.countsB = None
        self.resSeqsB = None


    @property
    def _res_namesA(self):
        return np.array([r.name for r in self.structA.top.residues])

    @property
    def _res_namesB(self):
        return np.array([r.name for r in self.structB.top.residues])

    @property
    def _resSeqsA(self):
        return np.array([r.resSeq for r in self.structA.top.residues])

    @property
    def _resSeqsB(self):
        return np.array([r.resSeq for r in self.structB.top.residues])

    @property
    def resi_contactsA(self):
        if self.resSeqsA is None:
            resi_contactsA = None
        else:
            resi_contactsA = ra.RaggedArray(
                [
                    np.unique(
                        [
                            np.where(self._resSeqsA == r)[0]
                            for r in resSeqsA_frame]).flatten()
                    for resSeqsA_frame in self.resSeqsA])
        return resi_contactsA

    @property
    def resi_contactsB(self):
        if self.resSeqsB is None:
            resi_contactsB = None
        else:
            resi_contactsB = ra.RaggedArray(
                [
                    np.unique(
                        [
                            np.where(self._resSeqsB == r)[0]
                            for r in resSeqsB_frame]).flatten()
                    for resSeqsB_frame in self.resSeqsB])
        return resi_contactsB

    @property
    def resSeq_contactsA(self):
        if self.resSeqsA is None:
            resSeq_contactsA = None
        else:
            resSeq_contactsA = ra.RaggedArray(
                [np.unique(resSeqs) for resSeqs in self.resSeqsA])
        return resSeq_contactsA

    @property
    def resSeq_contactsB(self):
        if self.resSeqsB is None:
            resSeq_contactsB = None
        else:
            resSeq_contactsB = ra.RaggedArray(
                [np.unique(resSeqs) for resSeqs in self.resSeqsB])
        return resSeq_contactsB


    def count_contacts(self, **kwargs):
        """Compute contacts between structA and structB.

        Inputs
        ----------
        structA : md.Trajectory,
            The first structure to use for calculating close contacts.
        structB : md.Trajectory,
            The second structure to use for calculating close contacts.
        criterion : str, choices=['VDW', 'dist'], default='VDW',
            Criterion for determining close contacts, either as a fraction
            of the van der Waals radii, or within a particular distance.
        VDW_dist : float, default=1.4,
            The fraction of van der Waals radii to use as a cutoff for
            determining close-contacts. i.e. considered a contact if a
            distance pair is less than 1.4x the sum of the van der Waals
            radii (default).
        dist_cutoff : float, default=0.39,
            The distance cutoff for determining a close-contact. Default is
            set to 0.39 nm.
        atoms : str, choices=['all', 'heavy'], default='all',
            Atoms to include for close contacts, 'all', or 'heavy'. Use all
            atoms or only heavy atoms.
        mode : str, choices=['residue', 'atom'], default='residue',
            Option for computing granularity of contacts. Can count based
            on residues or per atom pair.
        """
        self.pairs, self.counts = pairwise_contacts(self.structA, self.structB, **kwargs)

        # initialize variables
        uniqueA, countsA = [], []
        uniqueB, countsB = [], []
        res_namesA, resSeqsA, atomsA = [], [], []
        res_namesB, resSeqsB, atomsB = [], [], []

        # iterate through conformation contact info
        for n in np.arange(self.pairs.shape[0]):

            # find unique contact names for structures A and B
            uniqueA_tmp = np.unique(self.pairs[n][:,0])
            uniqueB_tmp = np.unique(self.pairs[n][:,1])

            # count the contacts for each unique name
            countsA_tmp = np.array(
                [
                    np.sum(self.counts[n][self.pairs[n][:,0] == u])
                    for u in uniqueA_tmp])
            countsB_tmp = np.array(
                [
                    np.sum(self.counts[n][self.pairs[n][:,1] == u])
                     for u in uniqueB_tmp])

            # if there are contacts, parse residue info (i.e.
            # ASN120-CA -> res_name = 'ASN', resSeq = 120, atom = 'CA')
            if uniqueA_tmp.shape[0] > 0:
                res_namesA_tmp, resSeqsA_tmp, atomsA_tmp = _parse_residues(
                    uniqueA_tmp)
                res_namesB_tmp, resSeqsB_tmp, atomsB_tmp = _parse_residues(
                    uniqueB_tmp)
            # if no contacts, return empty lists
            else:
                res_namesA_tmp, resSeqsA_tmp, atomsA_tmp = [],[],[]
                res_namesB_tmp, resSeqsB_tmp, atomsB_tmp = [],[],[]

            # sort based on residue numbers and append data to lists
            iisA = np.argsort(resSeqsA_tmp)
            if iisA.shape[0] > 0:
                uniqueA.append(uniqueA_tmp[iisA])
                countsA.append(
                    np.array(countsA_tmp[iisA], dtype=int))
                res_namesA.append(res_namesA_tmp[iisA])
                resSeqsA.append(
                    np.array(resSeqsA_tmp[iisA],dtype=int))
            else:
                uniqueA.append(uniqueA_tmp)
                countsA.append(
                    np.array(countsA_tmp, dtype=int))
                res_namesA.append(res_namesA_tmp)
                resSeqsA.append(
                    np.array(resSeqsA_tmp,dtype=int))

            iisB = np.argsort(resSeqsB_tmp)
            if iisB.shape[0] > 0:
                uniqueB.append(uniqueB_tmp[iisB])
                countsB.append(
                    np.array(countsB_tmp[iisB], dtype=int))
                res_namesB.append(res_namesB_tmp[iisB])
                resSeqsB.append(
                    np.array(resSeqsB_tmp[iisB], dtype=int))
            else:
                uniqueB.append(uniqueB_tmp)
                countsB.append(
                    np.array(countsB_tmp, dtype=int))
                res_namesB.append(res_namesB_tmp)
                resSeqsB.append(
                    np.array(resSeqsB_tmp,dtype=int))

            # try to sort atom names. if it doesnt work, its probably because
            # atoms don't exist (are `None`), so just add as is
            try:
                atomsA.append(atomsA_tmp[iisA])
                atomsB.append(atomsB_tmp[iisB])
            except:
                atomsA.append([atomsA_tmp]*iisA.shape[0])
                atomsB.append([atomsB_tmp]*iisB.shape[0])

        # make ragged arrays of each data item
        self.uniqueA = ra.RaggedArray(uniqueA)
        self.uniqueB = ra.RaggedArray(uniqueB)
        self.countsA = ra.RaggedArray(countsA)
        self.countsB = ra.RaggedArray(countsB)
        self.res_namesA = ra.RaggedArray(res_namesA)
        self.resSeqsA = ra.RaggedArray(resSeqsA)
        self.res_namesB = ra.RaggedArray(res_namesB)
        self.resSeqsB = ra.RaggedArray(resSeqsB)

        # try to make a ragged array of atoms
        # this will likely fail if there are no atoms AND some pairs have no contacts
        # so in that case, just add a bunch of `None`s
        try:
            self.atomsA = ra.RaggedArray(atomsA)
            self.atomsB = ra.RaggedArray(atomsB)
        except:
            self.atomsA = [None]*res_namesA.shape[0]#ra.RaggedArray(
              #  [None]*res_namesA._data.shape[0], lengths=res_namesA.lengths)
            self.atomsB = [None]*res_namesA.shape[0]#ra.RaggedArray(
               # [None]*res_namesB._data.shape[0], lengths=res_namesB.lengths)

        if np.all(None == self.atomsA._data):
            self.atomsA = np.array([None]*self.atomsA.shape[0])
        if np.all(None == self.atomsB._data):
            self.atomsB = np.array([None]*self.atomsB.shape[0])


    def __repr__(self):
        out = "Contacts(%s, %s)" % (self.nameA, self.nameB)
        return out

    def output(self, mode='table', residues='block', frames=0, **kwargs):
        """Output control for contacts
        
        Inputs
        ----------
        mode : str, choices=['table','fancy','chimerax','pymol']
            Output styles are either as a pandas table, a fancy string,
            or as a quick input to visualize in chimerax or pymol.
        residues : str, choices=['block','fancy','single'], default='block',
            Output control for representation of residues, i.e. TYR, Tyr, Y
            for block, fancy, and single, respectively.
        frames : array-like, default=1,
            Number of frames to report.
        """
        if type(frames) is int:
            frames = np.array([frames])
        
        # compute contacts if they don't exist
        if self.uniqueA is None or self.countsA is None:
            self.count_contacts(**kwargs)

        spacing  = '    '

        for frame in frames:
            print("Frame: %d" % frame)

            if self.uniqueA[frame].shape[0] == 0:
                print("%sNo contacts\n\n" % spacing)
                continue

            # assert proper choices
            assert residues in ['block', 'fancy', 'single']
    
            # update residue names from block to either fancy or single letter
            # i.e. TYR -> Tyr -> Y
            if residues == 'fancy':
                res_namesA = np.array([convert_map_fancy[l] for l in self.res_namesA[frame]])
                res_namesB = np.array([convert_map_fancy[l] for l in self.res_namesB[frame]])
            elif residues == 'single':
                res_namesA = np.array([convert_map_single[l] for l in self.res_namesA[frame]])
                res_namesB = np.array([convert_map_single[l] for l in self.res_namesB[frame]])
            else:
                res_namesA = self.res_namesA[frame]
                res_namesB = self.res_namesB[frame]
        
            # print contacts as a pandas table
            if mode == 'table':
                df0 = _table_df(
                    res_namesA, self.resSeqsA[frame], self.atomsA[frame], self.countsA[frame])
                df1 = _table_df(
                    res_namesB, self.resSeqsB[frame], self.atomsB[frame], self.countsB[frame])

                table_out = _add_spacing(
                    "".join(
                        [
                            "%s%s%s\n" % (color.BOLD, self.nameA, color.END),
                            repr(df0),
                            "\n\n%s%s%s\n" % (color.BOLD, self.nameB, color.END),
                            repr(df1)]),
                    spacing)
                print(table_out)

            # print contacts as a fancy list (Residue, number, num_contacts
            # -> i.e. Tyr64(4))
            elif mode == 'fancy':
                fancy_out0 = _fancy_df(
                    res_namesA, self.resSeqsA[frame], self.atomsA[frame], self.countsA[frame])
                fancy_out1 = _fancy_df(
                    res_namesB, self.resSeqsB[frame], self.atomsB[frame], self.countsB[frame])
            
                fancy_out = _add_spacing(
                    "".join(
                        [
                            "%s\n" % self.nameA,
                            fancy_out0,
                            "\n\n%s\n" % self.nameB,
                            fancy_out1]),
                    spacing)
                print(fancy_out)

            # print contacts in a form to easily copy and past into chimerax
            # (comma separated)
            elif mode == 'chimerax':
                chimerax_out = _add_spacing(
                    "".join(
                        [
                            "%s%s%s\n" % (color.BOLD, self.nameA, color.END),
                            ", ".join(np.array(self.resSeqsA[frame], dtype=str)),
                            "\n\n%s%s%s\n" % (color.BOLD, self.nameB, color.END),
                            ", ".join(np.array(self.resSeqsB[frame], dtype=str))]),
                    spacing)
                print(chimerax_out)
                
            # print contacts in a form to easily copy and paste into pymol
            # (separated with a +)
            elif mode == 'pymol':
                pymol_out = _add_spacing(
                    "".join(
                        [
                            "%s%s%s\n" % (color.BOLD, self.nameA, color.END),
                            "+".join(np.array(self.resSeqsA[frame], dtype=str)),
                            "\n\n%s%s%s\n" % (color.BOLD, self.nameB, color.END),
                            "+".join(np.array(self.resSeqsB[frame], dtype=str))]),
                    spacing)
                print(pymol_out)
            print("\n\n")
