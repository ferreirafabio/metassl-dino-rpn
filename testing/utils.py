import matplotlib.pyplot as plt
import numpy as np


def plot(images, labels=None, title=None, normalised=False, **imshow_kwargs):
    if not isinstance(images[0], list):
        # Make a 2d grid even if there's just 1 row
        images = [images]
    # Add a cartesian coordinate system:
    # - coordinates range from -1 to 1
    # - equivalent to the underlying STN procedure
    # - center of the image is the origin of coordinate system
    if normalised:
        imshow_kwargs['extent'] = [-1, 1, -1, 1]
        imshow_kwargs['origin'] = 'lower'

    num_rows = len(images)
    num_cols = len(images[0])
    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(images):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img), **imshow_kwargs)
            if labels:
                im_lbl = labels[row_idx + col_idx]
                ax.set_title(im_lbl)
            if normalised:
                # Set the limits of the x and y axes to center the coordinate system on the image
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                # Invert the y-axis
                ax.invert_yaxis()
                # Draw the horizontal and vertical axes
                ax.axhline(0, color='black', lw=1)
                ax.axvline(0, color='black', lw=1)
            else:
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    if title:
        plt.suptitle(title)
    plt.tight_layout()
    plt.show()
