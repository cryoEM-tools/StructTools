import itertools
import mdtraj as md
import numpy as np
import pandas
from collections import OrderedDict
from . import formatting




#####################################################################
#                            distances                              #
#####################################################################


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


def _process_struct(struct, selection, atom_indices):
    if atom_indices is None:
        if selection in ['all', 'heavy', 'minimal', 'alpha']:
            iis_struct = struct.top.select_atom_indices(selection)
        else:
            try:
                iis_struct = struct.top.select(selection)
            except:
                raise
    else:
        iis_struct = atom_indices
    return struct.atom_slice(iis_struct)


def pairwise_contacts(
        structA, structB, criterion='dist',
        dist_cutoff=0.39, VDW_dist=1.4,
        selA='heavy', selB='heavy', atom_indicesA=None,
        atom_indicesB=None, mode='residue'):
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
    dist_cutoff : float, default=0.39,
        The distance cutoff for determining a close-contact. Default is
        set to 0.39 nm.
    VDW_dist : float, default=1.4,
        The fraction of van der Waals radii to use as a cutoff for
        determining close-contacts. i.e. considered a contact if a
        distance pair is less than 1.4x the sum of the van der Waals
        radii (default).
    selA : str, default='heavy',
        MDTraj selection syntax for structure A contacts.
    selB : str, default='heavy',
        MDTraj selection syntax for structure B contacts.
    atom_indicesA : list, default=None,
        Atom indices of structA to include for contacts.
    atom_indicesB : list, default=None,
        Atom indices of structB to include for contacts.
    mode : str, choices=['residue', 'atom'], default='residue',
        Option for computing granularity of contacts. Can count based
        on residues or per atom pair.

    Returns
    ----------
    contacts_dict : dictionary,
        A dictionary containing information of contacts between structA
        and structB.
    """
    
    # slice outrelevant indices
    structA = _process_struct(structA, selA, atom_indicesA)
    structB = _process_struct(structB, selB, atom_indicesB)

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

    else:
        raise

    # if mode is atom, count all atom-atom contacts as 1 count
    if mode == 'atom':

        # obtain atom names and chainids
        names0 = np.array([str(a) for a in structA.top.atoms])
        names1 = np.array([str(a) for a in structB.top.atoms])
        chain_ids0 = np.array([a.residue.chain.index for a in structA.top.atoms])
        chain_ids1 = np.array([a.residue.chain.index for a in structB.top.atoms])

        # count contacts
        iis_contacts = np.where(contacts_mask > 0)
        n_contacts = contacts_mask[iis_contacts]*1

    # if mode is residue, compile all atom-atom contacts 
    elif mode == 'residue':

        # obtain residue names
        names0 = np.array([str(r) for r in structA.top.residues])
        names1 = np.array([str(r) for r in structB.top.residues])
        chain_ids0 = np.array([r.chain.index for r in structA.top.residues])
        chain_ids1 = np.array([r.chain.index for r in structB.top.residues])

        # obtain residue atom indices (used to compile atom-atom results)
        resi_iis_structA = [[a.index for a in r.atoms] for r in structA.top.residues]
        resi_iis_structB = [[a.index for a in r.atoms] for r in structB.top.residues]

        n_resisA = len(resi_iis_structA)
        n_resisB = len(resi_iis_structB)
        contacts_mask_residues = np.zeros(
            (structA.n_frames, n_resisA, n_resisB), dtype=int)

        # iterate through structA and structB atom indices and compile
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
    
    # make dictionary of results
    frames, indexA, indexB = iis_contacts
    contact_dict = OrderedDict()
    contact_dict['frame'] = frames
    contact_dict['namesA'] = names0[indexA]
    contact_dict['namesB'] = names1[indexB]
    contact_dict['chainsA'] = chain_ids0[indexA]
    contact_dict['chainsB'] = chain_ids1[indexB]
    contact_dict['n_contacts'] = n_contacts
    
    return contact_dict


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


#####################################################################
#                             SASA                                  #
#####################################################################


small_number = 1E-9
def _calc_sasa_apo_holo(structA, structB, mode='residue', **kwargs):
    """Calculates the solvent exposure of two structures and their
       differences when bound.

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
    Strength is reported in fraction of SASA change between apo-holo
    structures.

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


#####################################################################
#                             Reporting                             #
#####################################################################


class Contact():
    def __init__(self, nameA, nameB, chainA, chainB, n_contact):
        self.resA, self.resSeqA, self.atomA = formatting._process_str_name(nameA)
        self.resB, self.resSeqB, self.atomB = formatting._process_str_name(nameB)
        self.chainA = chainA
        self.chainB = chainB
        self.n_contact = n_contact
    
    def __repr__(self):
        reprA = 'chainid %d %s%s' % (self.chainA, self.resA, self.resSeqA)
        reprB = 'chainid %d %s%s' % (self.chainB, self.resB, self.resSeqB)
        if self.atomA is not None:
            reprA = '%s-%s' % (reprA, self.atomA)
            reprB = '%s-%s' % (reprB, self.atomB)
            output = 'Contact(%s :: %s)' % (reprA, reprB)
        else:
            output = 'Contact(%s :: %s (%d))' % (reprA, reprB, self.n_contact)
        return output
    
    def __str__(self):
        strA = 'chainid %d %s%s' % (self.chainA, self.resA, self.resSeqA)
        strB = 'chainid %d %s%s' % (self.chainB, self.resB, self.resSeqB)
        if self.atomA is not None:
            strA = '%s-%s' % (strA, self.atomA)
            strB = '%s-%s' % (strB, self.atomB)
        return '%s :: %s' % (strA, strB)


class ContactsFrame():
    def __init__(
            self, structA_name='structA', structB_name='structB',
            namesA=None, namesB=None,
            chainsA=None, chainsB=None, n_contacts=None,
            contacts_dict=None):
        self.structA_name = structA_name
        self.structB_name = structB_name
        self._clear()
        if contacts_dict is not None:
            self._contacts = contacts_dict
        elif namesA is not None:
            self._add_data(
                namesA, namesB, chainsA, chainsB, n_contacts)
        self._update_unique_contacts()
        return
    
    @property
    def _contact_keys(self):
        return [k for k in self._contacts.keys()]
    
    @property
    def _last_contact_num(self):
        keys = self._contact_keys
        if len(keys) == 0:
            return -1
        else:
            return max(keys)
    
    @property
    def unique_resSeqsA(self):
        return np.unique(self._unique_contacts_resSeqsA)
    
    @property
    def unique_resSeqsB(self):
        return np.unique(self._unique_contacts_resSeqsB)

    def _clear(self):
        self._chain_namesA = None
        self._chain_namesB = None
        self._contacts = OrderedDict()
    
    def _add_data(
            self, namesA, namesB, chainsA, chainsB, n_contacts):
        assert len(namesA) == len(namesB)
        assert len(namesA) == len(chainsA)
        assert len(namesA) == len(chainsB)
        assert len(namesA) == len(n_contacts)
        
        contact_num = self._last_contact_num + 1
        for n in np.arange(len(namesA)):
            self._contacts[contact_num] = Contact(
                nameA=namesA[n], nameB=namesB[n],
                chainA=chainsA[n], chainB=chainsB[n],
                n_contact=n_contacts[n])
            contact_num += 1
        return
    
    def __getitem__(self, iis):
        if type(iis) is int:
            return self._contacts[iis]
        else:
            keys = np.array(self._contact_keys)
            new_contacts = OrderedDict()
            for n,k in enumerate(keys[iis]):
                new_contacts[n] = self._contacts[k]
            return ContactsFrame(contacts_dict=new_contacts)
    
    def __repr__(self):
        if self._last_contact_num >= 0:
            if self._contacts[0].atomA is None:
                mode = 'residue'
            else:
                mode = 'atom'
        else:
            mode = 'uninitiated'
        output = 'ContactsFrame(n_items=%d, mode=%s)' % \
            (self._last_contact_num + 1, mode)
        return output
    
    def __str__(self):
        return self._contacts.__str__()
    
    def __iter__(self):
        return iter(self._contacts.values())

    def _update_unique_contacts(self):
        contactsA_str, contactsB_str = np.array(
            [contact.__str__().split(" :: ") for contact in self]).T
        
        self._unique_contactsA_str = np.unique(contactsA_str)
        self._unique_countsA = [
            np.sum(contactsA_str == c) for c in self._unique_contactsA_str]
        
        self._unique_contactsB_str = np.unique(contactsB_str)
        self._unique_countsB = [
            np.sum(contactsB_str == c) for c in self._unique_contactsB_str]

        self._unique_contacts_chainsA, self._unique_contacts_resSeqsA, \
                self._unique_contacts_resA, self._unique_contacts_atomA = \
            formatting._data_from_str(self._unique_contactsA_str)
        self._unique_contacts_chainsB, self._unique_contacts_resSeqsB, \
                self._unique_contacts_resB, self._unique_contacts_atomB = \
            formatting._data_from_str(self._unique_contactsB_str)
        
    def set_chain_names(self, chain_namesA, chain_namesB):
        self._chain_namesA = chain_namesA
        self._chain_namesB = chain_namesB
        return
    
    def output(self, mode='table', style='block', **kwargs):
        output = formatting._output(
            self.structA_name, self._unique_contacts_chainsA,
            self._unique_contacts_resA, self._unique_contacts_resSeqsA,
            self._unique_contacts_atomA, self._unique_countsA,
            self.structB_name, self._unique_contacts_chainsB,
            self._unique_contacts_resB, self._unique_contacts_resSeqsB,
            self._unique_contacts_atomB, self._unique_countsB,
            style=style, mode=mode, chainA_names=self._chain_namesA,
            chainB_names=self._chain_namesB, **kwargs)
        return output


class Contacts():
    def __init__(
            self, structA_name='structA', structB_name='structB',
            contact_dict=None, contact_frames_dict=None):
        self._clear()
        if contact_dict is not None:
            self._parse_contact_dict(contact_dict)
            if contact_frames_dict is not None:
                raise
        elif contact_frames_dict is not None:
            self._contacts_frames = contact_frames_dict
    
    @property
    def _contact_frames_keys(self):
        return [k for k in self._contacts_frames.keys()]
    
    @property
    def unique_resSeqsA(self):
        resSeqs = np.unique(
            np.concatenate(
                [c.unique_resSeqsA for c in self._contacts_frames.values()]))
        return resSeqs
    
    @property
    def unique_resSeqsB(self):
        resSeqs = np.unique(
            np.concatenate(
                [c.unique_resSeqsB for c in self._contacts_frames.values()]))
        return resSeqs
    
    def set_chain_names(self, chain_namesA, chain_namesB):
        self._chain_namesA = chain_namesA
        self._chain_namesB = chain_namesB
        for c in self._contacts_frames.values():
            c.set_chain_names(chain_namesA, chain_namesB)
        return
    
    def _update_unique_contacts(self):
        for c in self._contacts_frames.values():
            c._update_unique_contacts()
        return
    
    def _clear(self):
        self._chain_namesA = None
        self._chain_namesB = None
        self._contacts_frames = OrderedDict()
        
    def _parse_contact_dict(self, contact_dict):
        self._clear()
        unique_frames = np.unique(contact_dict['frame'])
        for frame in unique_frames:
            iis = np.where(contact_dict['frame'] == frame)
            self._contacts_frames[frame] = \
                ContactsFrame(
                    namesA=contact_dict['namesA'][iis],
                    namesB=contact_dict['namesB'][iis],
                    chainsA=contact_dict['chainsA'][iis],
                    chainsB=contact_dict['chainsB'][iis],
                    n_contacts=contact_dict['n_contacts'][iis])
    
    def __iter__(self):
        return iter(self._contacts_frames.values())

    def __getitem__(self, iis):
        keys = np.array(self._contact_frames_keys)
        if type(iis) is int:
            return self._contacts_frames[keys[iis]]
        else:
            keys = np.array(self._contact_frames_keys)
            new_contacts = OrderedDict()
            for n,k in enumerate(keys[iis]):
                new_contacts[n] = self._contacts_frames[k]
            return Contacts(contact_frames_dict=new_contacts)
    
    @property
    def n_frames(self):
        return len(self._contact_frames_keys)
    
    def __repr__(self):
        output = 'Contact(n_frames=%d)' % self.n_frames
        return output
    
    def output(self, frame=0, **kwargs):
        return self._contacts_frames[frame].output(**kwargs)
