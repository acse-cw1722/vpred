# helper functions
import matplotlib.pyplot as plt

ROW = 4
COL = 4
NUM_FRAME = 16


def show_frames(vid):
    fig, axs = plt.subplots(ROW, COL)
    count = 0
    for r in range(ROW):
        for c in range(COL):
            axs[r, c].imshow(vid[count])
            count += 1
    return fig
