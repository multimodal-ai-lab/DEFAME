from pathlib import Path
from typing import Sequence

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

from defame.common.label import Label

COLOR_PALETTE = {
    "light-gray": "#DDDDDD",
    "mid-gray": "#888888",
    "dark-gray": "#333333",
    "1b": "#005AA9",  # dark-blue
    "dark-blue": "#005AA9",
    "2b": "#0083CC",  # blue
    "blue": "#0083CC",
    "2a": "#009CDA",  # light-blue
    "light-blue": "#009CDA",
    "3b": "#009D81",  # turquoise
    "turquoise": "#009D81",
    "8b": "#EC6500",  # orange
    "orange": "#EC6500",
    "7b": "#F5A300",  # yellow-orange
    "yellow-orange": "#F5A300",
}


def plot_confusion_matrix(predictions: Sequence[Label],
                          ground_truth: Sequence[Label],
                          classes: Sequence[Label],
                          benchmark_name: str,
                          save_dir: Path = None):
    """Generates and shows a confusion matrix for the given predictions and ground truth. The order of the classes
    displayed follows the order of the `classes` parameter."""

    class_indices = {c: v for v, c in enumerate(classes)}

    # Construct confusion matrix
    confusion_matrix = np.zeros((len(classes), len(classes)), dtype="float")
    for pred, gt in zip(predictions, ground_truth):
        if isinstance(pred, str):
            pred = Label[pred]
            gt = Label[gt]
        if pred != Label.REFUSED_TO_ANSWER and (gt in classes) and (pred in classes):
            confusion_matrix[class_indices[gt], class_indices[pred]] += 1

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
        plt.savefig(save_dir / "confusion.pdf")
        plt.savefig(save_dir / "confusion.png")
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

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
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
        threshold = im.norm(data.max()) / 2.

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


def plot_grouped_bar_chart(x_labels: list,
                           values: dict,
                           title: str,
                           x_label: str,
                           y_label: str,
                           show_values: bool = True,
                           colors: list = None,
                           save_path: str = None):
    if colors is not None:
        assert len(colors) == len(values)

    x = np.arange(len(x_labels))  # the label locations
    width = 0.15  # the width of the bars
    multiplier = 0

    fig, ax = plt.subplots(layout='constrained')

    for i, (attribute, measurement) in enumerate(values.items()):
        offset = width * multiplier
        color = colors[i] if colors is not None else None
        rects = ax.bar(x + offset, measurement, width, label=attribute, color=color)
        if show_values:
            ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    ax.set_xticks(x + width, x_labels)
    ax.legend()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()


def plot_histogram_comparison(data_rows: list,
                              title: str,
                              labels: list[str],
                              y_label: str,
                              x_label: str = None,
                              n_bins: int = 20,
                              h_line_at: float = None,
                              secondary_labels: list[str] = None,
                              hist_range: (float, float) = None,
                              colors: list = None,
                              save_path: str = None):
    n_data_rows = len(data_rows)
    assert len(labels) == n_data_rows
    if colors is not None:
        assert len(colors) == n_data_rows

    data_rows = np.array(data_rows)

    if hist_range is None:
        hist_range = (np.min(data_rows), np.max(data_rows))

    binned_data_sets = [
        np.histogram(d, range=hist_range, bins=n_bins)[0]
        for d in data_rows
    ]
    binned_maximums = np.max(binned_data_sets, axis=1)
    x_locations = np.arange(0, sum(binned_maximums), np.max(binned_maximums))

    # The bin_edges are the same for all the histograms
    bin_edges = np.linspace(hist_range[0], hist_range[1], n_bins + 1)
    heights = np.diff(bin_edges)
    centers = bin_edges[:-1] + heights / 2

    # Cycle through and plot each histogram
    fig, ax = plt.subplots(layout="constrained")
    for i, (x_loc, binned_data) in enumerate(zip(x_locations, binned_data_sets)):
        color = colors[i] if colors is not None else None
        lefts = x_loc - 0.5 * binned_data
        ax.barh(centers, binned_data, height=heights, left=lefts, color=color)

    # Plot horizontal line
    if h_line_at is not None:
        plt.axhline(y=h_line_at, linestyle='dashed', color='gray')

    # Plot higher-level labels
    if secondary_labels is not None:
        ticks = ax.get_xticks()
        left = ticks[1]
        right = ticks[-1]
        width = right - left
        n_secondary_labels = len(secondary_labels)
        distance = width // n_secondary_labels
        secondary_ax = ax.secondary_xaxis(location=-0.06)
        secondary_ax.tick_params(axis=u'both', which=u'both', length=0)
        secondary_ax.spines[['bottom']].set_visible(False)
        secondary_ax.set_xticks([(i + 0.25) * distance for i in range(n_secondary_labels)], labels=secondary_labels)

    ax.set_xticks(x_locations, labels)

    ax.set_title(title)
    ax.set_ylabel(y_label)
    if x_label is not None:
        ax.set_xlabel(x_label)
    # plt.xticks(rotation=45)
    # plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    plt.show()
