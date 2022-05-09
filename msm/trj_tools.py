import mdtraj as md
import numpy as np

def backtrack_assigns(assigns, start_state, final_state):
    state_to_backtrack = final_state
    max_n_iters = assigns.shape[0]
    trj_nums = []
    frame_nums = []
    n_iter = 0
    while (state_to_backtrack != start_state) and (n_iter <= max_n_iters):
        iis = np.where(assigns == state_to_backtrack)
        trj_nums.append(iis[0][0])
        frame_num = iis[1][np.where(iis[0] == iis[0][0])]
        frame_nums.append(frame_num[-1])
        state_to_backtrack = assigns[trj_nums[-1], 0]
        n_iter += 1
    if n_iter <= max_n_iters:
        iis = np.where(assigns == state_to_backtrack)
        trj_nums.append(iis[0][0])
        frame_num = iis[1][np.where(iis[0] == iis[0][0])]
        frame_nums.append(frame_num[-1])
        state_to_backtrack = assigns[trj_nums[-1], 0]
    return trj_nums[::-1], frame_nums[::-1]


def gen_continuous_trj(trj_filenames, trj_nums, frame_nums, top):
    trjs = []
    for f,n in zip(trj_filenames[trj_nums], frame_nums):
        trjs.append(md.load(f, top=top)[:n])
    trjs = md.join(trjs)
    return trjs
