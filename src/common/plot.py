from typing import Sequence

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from common.label import Label


def plot_confusion_matrix(predictions: Sequence[Label],
                          ground_truth: Sequence[Label],
                          classes: Sequence[Label],
                          benchmark_name: str,
                          save_dir: str = None):
    class_conversion = {c: v for v, c in enumerate(classes)}

    # Construct confusion matrix
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype="float")
    for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        confusion_matrix[class_conversion[gt], class_conversion[pred]] += 1

    correct = np.copy(confusion_matrix)
    wrong = np.copy(confusion_matrix)
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            if i == j:
                wrong[i, j] = np.nan
            else:
                correct[i, j] = np.nan

    # Plot confusion matrix
    fig, ax = plt.subplots()
    class_names = [c.name for c in classes]
    v_max = np.max(len(ground_truth) // 3)
    hm, _ = heatmap(correct, class_names, class_names, cmap="Greens", show_cbar=False, ax=ax, vmin=0, vmax=v_max)
    annotate_heatmap(hm, valfmt="{x:.0f}")
    hm, _ = heatmap(wrong, class_names, class_names, cmap="Reds", show_cbar=False, ax=ax, vmin=0, vmax=v_max)
    annotate_heatmap(hm, valfmt="{x:.0f}")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title(f"{benchmark_name} Confusion Matrix")
    fig.tight_layout()
    if save_dir:
        plt.savefig(save_dir + "confusion.pdf")
        plt.savefig(save_dir + "confusion.png")
    plt.show()


def heatmap(data, row_labels, col_labels, show_cbar=True, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    if show_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = mpl.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not np.ma.is_masked(data[i, j]):
                kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
                text = im.axes.text(j, i, valfmt(data[i, j]), **kw)
                texts.append(text)

    return texts
