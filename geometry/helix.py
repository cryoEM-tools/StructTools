import numpy as np
import mdtraj as md


def __avg_vecs(points_to_avg):
    """helper to average forward facing vectors from points"""
    avg = np.array(
        [
            points_to_avg[:,i+1] - points_to_avg[:,i]
            for i in np.arange(points_to_avg.shape[1]-1)]).mean(axis=0)
    return avg


def _get_backbone_nums(top, resnums):
    """Returns the atom indices that correspond to N, CA, and C for a
    list a residues."""
    backbone_nums = np.concatenate(
        [
            [
                top.select("resSeq %d and name N" % res)[0],
                top.select("resSeq %d and name CA" % res)[0],
                top.select("resSeq %d and name C" % res)[0]]
            for res in np.sort(resnums)])
    return backbone_nums


def thread(
        trj, helix_resnums, helix_start=None, helix_stop=None):
    """Calculates a thread along the center of a helix, with center
    point and vector corresponding to each residue location. Computes
    over every frame in a trajectory. Grabs residues 2 residues from
    front and back to perform calculation.

    Parameters
    ----------
    trj : md.Trajectory object
        An MDTraj trajectory object containing frames of structures to
        compute helix-vectors from.
    helix_resnums : array, shape [n_residues, ], optional, default: None
        A list of residues that correspond to an alpha-helix. This is
        useful if residue numbers within a helix are unordinary. If a
        list of residues is not supplied, a start and stop residue can
        be specified.
    helix_start : int, optional, default: None
        The starting residue of the helix.
    helix_start : int, optional, default: None
        The ending residue of the helix.

    Returns
    ----------
    vectors : array, [n_frames, n_residues, 3]
        A list of unit-vectors corresponding to the direction of the
        (residue-local) alpha-helix for each frame in the trajectory.
    center_coords : array, [n_frames, n_residues, 3]
        Center of the helical thread (residue-local).
    """
    if (helix_resnums is None) and ((helix_start is None) or
                                    (helix_end is None)):
        raise exception.ImproperlyConfigured(
            "Either 'helix_resnums' or 'helix_start' and 'helix_end' "
            "are required.")
    elif helix_resnums is None:
        helix_resnums = np.arange(helix_start, helix_end+1)
    
    # pad residue numbers (2 in back, 2 in front)
    helix_resnums = np.concatenate(
        [
            np.arange(helix_resnums[0]-2, helix_resnums[0]),
            helix_resnums,
            np.arange(helix_resnums[-1]+1, helix_resnums[-1]+3)])

    # extract backbone coordinates (N,CA,C)
    top = trj.topology
    backbone_nums = _get_backbone_nums(top, helix_resnums)
    backbone_coords = trj.xyz[:, backbone_nums]    

    # average 4 residue windows to obtain center thread
    n_sliding_window = 12
    backbone_avgs = np.transpose(
        [
            backbone_coords[:,n:n+n_sliding_window].mean(axis=1)
            for n in np.arange(backbone_coords.shape[1],step=3)],
        axes=(1,0,2))
    center_coords = backbone_avgs[:,1:-3]

    # compute vectors pointing along thread and average 4 to obtain
    # per residue directionality
    n_vec_avg = 4
    vectors = np.transpose(
        [
            __avg_vecs(backbone_avgs[:,n:n+n_vec_avg])
            for n in np.arange(backbone_avgs.shape[1]-n_vec_avg)],
        axes=(1,0,2))

    # unit norm vectors
    vector_mags = np.sqrt(np.einsum('ijk,ijk->ij',vectors, vectors))[:,:,None]
    vectors_normed = vectors / vector_mags 

    return vectors_normed, center_coords


def local_bending(trj, helix_resnums, n_res_buffer=2, **kwargs):
    """Calculates the local bending angle of an alpha-helix on a per-residue
    basis, in units of degrees

    Inputs
    ----------
    trj : mdtraj.Trajectory,
        A list of structures to use for calculating helix bending angles.
    helix_resnums : array-like, shape=(n_residues, )
        A list of resSeq numbers that correspond to a helix to use for
        calculate bending angles. Will extend 2 residues in both directions
        for local helix directions.
    n_res_buffer : int, default=2,
        The number of residues in each direction to consider "local" bending.
        For a value of 2, calculates angle between i-2 and i+2 helical
        direction vectors.
        
    Outputs
    ----------
    bend_angles : nd.array, shape=(n_frames, n_residues, 3),
        A list of helical bending angles per residue for each frame.

    """
    # determine localized helix directions
    vectors_normed, _ = thread(trj, helix_resnums, **kwargs)

    # define window for measuring angles
    window = 2*n_res_buffer

    # determine bending angles
    bend_angles = np.zeros(shape=(vectors_normed.shape[0], vectors_normed.shape[1]))
    bend_angles_tmp = np.arccos(
        np.einsum(
            'ijk,ijk->ij', vectors_normed[:,window:], vectors_normed[:,:-window]))
    bend_angles_tmp *= 360/(np.pi*2)

    # angles of residues at ends is 0 (excluded from window size)
    bend_angles[:,n_res_buffer:-n_res_buffer] = bend_angles_tmp
    return bend_angles
