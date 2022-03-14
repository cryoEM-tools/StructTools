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


def helix_thread(
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

    return vectors, center_coords
