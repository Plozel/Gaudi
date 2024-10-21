import os
from typing import List, Optional, Union, Tuple, Any, Dict

import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, to_rgb

import pandas as pd

from scipy.interpolate import splprep, splev
from scipy.sparse import issparse
from scipy.spatial.distance import cdist
from skimage.color import lab2rgb, rgb2lab

from anndata import AnnData
import squidpy as sq
import scanpy as sc



def get_colormap_for_labels(n_colors, seed=1, listed_colormap=False):
    """
    Generates a colormap with a fixed number of distinct colors.

    Args:
        n_colors (int): The number of distinct colors to generate.
        seed (int): Random seed for consistent color generation.

    Returns:
        A colormap object with the specified number of distinct colors.
    """
    np.random.seed(seed)  # Set the seed for reproducibility
    # Start with the tab20 colors
    base_colors = [to_rgb(c) for c in plt.get_cmap("tab20").colors]
    if n_colors <= len(base_colors):
        colors = base_colors[:n_colors]
    else:
        # Convert base colors to LAB for better distance comparisons
        lab_colors = rgb2lab(np.array([base_colors])).reshape(-1, 3)
        colors = lab_colors

        # Generate new colors
        while len(colors) < n_colors:
            best_new_color = None
            max_dist = 0
            for _ in range(100):  # Reduced number of iterations
                new_color = np.random.rand(3)
                new_color_lab = rgb2lab(new_color[None, None, :]).reshape(-1, 3)
                min_dist = np.min(cdist(colors, new_color_lab))
                if min_dist > max_dist:
                    max_dist = min_dist
                    best_new_color = new_color_lab

            colors = np.vstack([colors, best_new_color])

        # Convert LAB colors back to RGB
        colors = lab2rgb(colors.reshape(-1, 1, 3)).reshape(-1, 3)

    colormap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=n_colors)
    if listed_colormap:
        colormap = ListedColormap(colors)

    return colormap


def convert_to_listed_colormap(linear_colormap, n_colors=256):
    """Convert a LinearSegmentedColormap to a ListedColormap."""

    colors = linear_colormap(np.linspace(0, 1, n_colors))
    listed_colormap = ListedColormap(colors)
    return listed_colormap


def plot_communities(
    communities_coordinates: List[np.ndarray],
    points: Optional[np.ndarray] = None,
    labels: Optional[Union[np.ndarray, List[Union[int, float, str]]]] = None,
    subset_labels: Optional[Union[np.ndarray, List[Union[int, float, str]]]] = None,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
    invert_yaxis: bool = False,
    invert_xaxis: bool = False,
    standard_scale: Optional[str] = None,
    **kwargs,
) -> None:
    """Plots communities with optional points and labels.

    Args:
        communities_coordinates: List of coordinates for each community.
        points: Optional array of points to plot.
        labels: Optional array or list of labels for the communities.
        subset_labels: Optional subset of labels for highlighting specific communities.
        title: Optional title for the plot.
        save_path: Optional path to save the plot.
        invert_yaxis: Flag to invert the y-axis.
        invert_xaxis: Flag to invert the x-axis.
        standard_scale: Optional standard scaling method ('log' or 'obs').
        **kwargs: Additional style options.
    """
    default_style_options = {
        "figsize": (10, 12),
        "file_format": "png",
        "dpi": 300,
        "cmap_continuous": "viridis",
        "cmap_discrete": "tab20",
        "show_legend": True,
        "line_width": 0.3,
        "point_size": 0.1,
        "alpha": 1.0,
        "background_color": "white",
        "legend_loc": "upper left",
        "legend_bbox_to_anchor": (1, 1),
        "color_bar_label": None,
        "vmin": None,
        "vmax": None,
    }

    style_options = {**default_style_options, **kwargs}
    fig, ax = plt.subplots(figsize=style_options["figsize"])
    ax.set_facecolor(style_options["background_color"])
    fig.set_facecolor(style_options["background_color"])

    subset_labels = np.array(subset_labels) if subset_labels is not None else None
    labels = labels.toarray().squeeze() if issparse(labels) else labels
    is_continuous = (
        np.issubdtype(np.array(labels).dtype, np.floating)
        if labels is not None
        else False
    )
    labels = np.array(labels) if labels is not None else None
    original_labels = np.unique(labels) if labels is not None else None
    filtered_indices = [
        i for i, region in enumerate(communities_coordinates) if len(region) >= 1
    ]
    communities_coordinates = [communities_coordinates[i] for i in filtered_indices]
    labels = labels[filtered_indices] if labels is not None else None

    # Processing labels and applying scaling if necessary
    label_to_color_index, colors, unique_labels = process_labels(
        labels,
        is_continuous,
        subset_labels,
        original_labels,
        standard_scale,
        style_options,
        communities_coordinates,
    )

    plot_regions(communities_coordinates, colors, style_options, ax)

    add_plot_annotations(is_continuous, labels, unique_labels, style_options, ax, label_to_color_index)

    if title:
        plt.title(title)

    if points is not None:
        plt.scatter(
            points[:, 0],
            points[:, 1],
            c="black",
            s=style_options["point_size"],
            zorder=100000000000,
        )

    plt.tight_layout()

    if invert_yaxis:
        ax.invert_yaxis()

    if invert_xaxis:
        ax.invert_xaxis()
        
    if save_path:
        save_plot(fig, ax, save_path, style_options)

    plt.show()


def process_labels(
    labels,
    is_continuous,
    subset_labels,
    original_labels,
    standard_scale,
    style_options,
    communities_coordinates,
):

    if labels is not None:
        if is_continuous:
            labels = apply_standard_scale(labels, standard_scale)
            vmin, vmax, norm, cmap = configure_continuous_colormap(
                labels, style_options
            )
            colors = cmap(norm(labels))
            label_to_color_index = None
        else:
            unique_labels = np.unique(labels)
            if len(unique_labels) > 20:  # Example threshold
                cmap = get_colormap_for_labels(len(unique_labels))
            else:
                cmap = plt.get_cmap(style_options["cmap_discrete"])

            label_to_color_index = {
                label: index for index, label in enumerate(original_labels)
            }
            colors = [
                cmap(label_to_color_index.get(label, 0) % cmap.N) for label in labels
            ]

            if subset_labels is not None:
                colors = [
                    (
                        cmap(label_to_color_index.get(label, 0) % cmap.N)
                        if label in subset_labels
                        else (0.95, 0.95, 0.95, 1)
                    )
                    for label in labels
                ]
    else:
        colors = [style_options["background_color"]] * len(communities_coordinates)

    return label_to_color_index, colors, unique_labels if "unique_labels" in locals() else None


def plot_regions(communities_coordinates, colors, style_options, ax):
    base_zorder = 1  # Starting zorder for regions

    for index, region in enumerate(communities_coordinates):
        color = colors[index]
        if (region != region[0, :]).all():
            region = np.vstack([region, region[0, :]])

        try:
            tck, _ = splprep(region.T, s=0)
            unew = np.linspace(0, 1.0, 100)
            out = splev(unew, tck)
            out[0] = np.append(out[0], out[0][0])
            out[1] = np.append(out[1], out[1][0])

            # Set the zorder dynamically
            ax.fill(out[0], out[1], alpha=style_options["alpha"], color=color, zorder=base_zorder)
            ax.plot(out[0], out[1], color="black", linewidth=style_options["line_width"], zorder=base_zorder + 1)
        except Exception as e:
            continue  # Skip regions that cannot be processed
        
        # Increment the base_zorder for the next region
        base_zorder += 2


def add_plot_annotations(is_continuous, labels, unique_labels, style_options, ax, label_to_color_index):
    """Adds legends for discrete labels or color bars for continuous labels to the plot.

    Args:
        is_continuous (bool): Indicates if the labels are continuous.
        labels (np.ndarray): Label data.
        unique_labels (np.ndarray): Unique labels, for discrete labels only.
        style_options (dict): A dictionary of style options.
        ax (matplotlib.axes.Axes): The plot axes to which the annotations are added.
    """
    if labels is not None:
        if not is_continuous and style_options.get("show_legend", False):

            if len(unique_labels) > 20:  # Example threshold
                cmap = get_colormap_for_labels(len(unique_labels))
            else:
                cmap = plt.get_cmap(style_options["cmap_discrete"])

            legend_elements = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=label,
                    markerfacecolor=cmap(label_to_color_index[label]),
                    markersize=10,
                )
                for i, label in enumerate(unique_labels)
            ]

            ncol = int(len(unique_labels) / 10) if len(unique_labels) > 20 else 1

            ax.legend(
                legend_elements,
                unique_labels,
                title="Labels",
                loc=style_options["legend_loc"],
                bbox_to_anchor=style_options["legend_bbox_to_anchor"],
                ncol=ncol,
            )
        elif is_continuous:
            if ax is None:
                ax = plt.gca() 

            cmap = style_options.get("cmap")
            vmin = style_options.get("vmin")
            vmax = style_options.get("vmax")

            if vmin is None:
                vmin = np.min(labels) 
            if vmax is None:
                vmax = np.max(labels) 

            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
            scalar_mappable.set_clim(vmin=vmin, vmax=vmax)
            scalar_mappable.set_array([])

            if ax.collections:
                for coll in ax.collections:
                    coll.remove()

            cbar = plt.colorbar(scalar_mappable, ax=ax)
            if style_options.get("color_bar_label"):
                cbar.set_label(style_options["color_bar_label"])


def save_plot(fig, ax, save_path, style_options):
    fig.savefig(
        f"{save_path}.{style_options['file_format']}",
        format=style_options["file_format"],
        dpi=style_options["dpi"],
    )
    # Reset background before saving in a different format if needed
    ax.set_facecolor("white")
    fig.set_facecolor("white")
    fig.savefig(f"{save_path}.pdf", format="pdf", dpi=style_options["dpi"])


def apply_standard_scale(
    labels: np.ndarray, standard_scale: Optional[str]
) -> np.ndarray:
    "Applies standard scaling to labels based on the method specified."
    if standard_scale == "log":
        labels = np.log1p(labels)
    elif standard_scale == "obs":
        # Scale the labels across observations globally
        labels_min, labels_max = np.min(labels), np.max(labels)
        labels = (labels - labels_min) / (labels_max - labels_min)
    return labels


def configure_continuous_colormap(labels: np.ndarray, style_options: dict):
    """Configures colormap and normalization for continuous labels.

    Args:
        labels: Continuous labels.
        style_options: A dictionary of style options.

    Returns:
        A tuple containing vmin, vmax, a Normalize instance, and a colormap.
    """
    vmin = (
        style_options["vmin"] if style_options["vmin"] is not None else np.min(labels)
    )
    vmax = (
        style_options["vmax"] if style_options["vmax"] is not None else np.max(labels)
    )
    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.cm.get_cmap(style_options["cmap_continuous"])
    return vmin, vmax, norm, cmap


def plot_dots(
    adata,
    color: List[str],
    layer: str = None,
    size: int = None,
    figsize: Tuple[int, int] = None,
    title: str = None,
    dpi: int = None,
    save_path: str = None,
    palette: str = None,
    cmap: str = "viridis",
    invert_yaxis: bool = False,
    invert_xaxis: bool = False,
    **kwargs: Any,
):

    if palette is None:
        if color in adata.obs.columns:
            n_colors = adata.obs[color].nunique()
            palette = get_colormap_for_labels(n_colors, listed_colormap=True)
        else:
            palette = cmap

    ax = sq.pl.spatial_scatter(
        adata,
        color=color,
        size=size,
        layer=layer,
        figsize=figsize,
        palette=palette,
        cmap=cmap,
        shape=None,
        return_ax=True,
        dpi=dpi,
        title=title,
        **kwargs,
    )
    if invert_xaxis:
        ax.invert_xaxis()
    if invert_yaxis:
        ax.invert_yaxis()

    if save_path is not None:
        ax.figure.savefig(save_path, dpi=dpi)


def plot_composition(
    group_2_composition_per_group_1,
    group_1,
    group_2,
    save_path: Optional[str] = None,
    custom_group_1_labels: Optional[Dict[int, str]] = None,
    custom_group_2_labels: Optional[Dict[int, str]] = None,
    **kwargs,
):

    default_style_options = {
        "dpi": 100,
        "figsize": (12, 8),
        "background_color": "white",
        "file_format": "png",
        "title": f"{group_2} composition within {group_1}",
        "x_label_rotation": 30,
    }

    style_options = {**default_style_options, **kwargs}

    # Apply custom x-axis labels if provided
    if custom_group_1_labels:
        group_2_composition_per_group_1.rename(index=custom_group_1_labels, inplace=True)

    fig, ax = plt.subplots(dpi=style_options["dpi"], figsize=style_options["figsize"])
    ax.set_facecolor(style_options["background_color"])
    fig.set_facecolor(style_options["background_color"])

    # Get a colormap for the labels
    n_colors = len(group_2_composition_per_group_1.columns)
    colormap = get_colormap_for_labels(n_colors, listed_colormap=True)
    colors = [colormap(i) for i in range(n_colors)]

    # Plotting
    group_2_composition_per_group_1.plot(kind="bar", stacked=True, ax=ax, color=colors)

    ax.set_title(style_options["title"])
    ax.set_xlabel("Context Clusters")
    ax.set_ylabel("Proportion")

    ax.set_xticklabels(ax.get_xticklabels(), rotation=style_options["x_label_rotation"])

    # Apply custom cell type names if provided
    if custom_group_2_labels:
        legend_labels = [
            custom_group_2_labels.get(i, celltype)
            for i, celltype in enumerate(group_2_composition_per_group_1.columns)
        ]
        ax.legend(legend_labels, title=group_2, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.legend(title=group_2, bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()

    if save_path:
        save_plot(fig, ax, save_path, style_options)


def _get_valid_labels(adata, groupby, min_counts=1, exclude_labels=None):

    labels = adata.obs[groupby]

    unique_labels, counts = np.unique(labels, return_counts=True)
    less_than_min_counts_labels = set(unique_labels[counts <= min_counts])
    unique_labels = set(unique_labels)
    valid_labels = unique_labels.difference(less_than_min_counts_labels)
    if exclude_labels is not None:
        valid_labels = valid_labels.difference(exclude_labels)
        print(
            f"Excluded labels: {exclude_labels}. Reason: These labels were excluded by the user."
        )

    if len(less_than_min_counts_labels) > 0:
        print(
            f"Excluded labels: {less_than_min_counts_labels}. Reason: These labels represent clusters with less than {min_counts + 1} observations."
        )

    return valid_labels


def plot_dotplot(
    adata: AnnData,
    var_names: int,
    groupby: str,
    groups="all",
    figsize=(25, 5),
    standard_scale="var",
    min_counts=1,
    exclude_labels=None,
    dendrogram=True,
    grid=True,
    cmap="Reds",
    largest_dot=200,
    legend_width=3,
    save_path=None,
    **kwargs,
) -> None:

    adata = adata.copy()
    if issparse(adata.X):
        adata.X = adata.X.todense()

    valid_labels = _get_valid_labels(adata, groupby, min_counts, exclude_labels)

    if groups != "all":
        groups = set(groups).intersection(valid_labels)
    else:
        groups = valid_labels

    adata = adata[adata.obs[groupby].isin(groups)]

    try:
        missing_groups_in_var_names = groups.difference(set(var_names.keys()))
        for group in missing_groups_in_var_names:
            var_names[group] = ["dummy_var"]

        adata.obs["dummy_var"] = 0

    except:
        pass

    sc.tl.dendrogram(adata, groupby=groupby, var_names=var_names)
    fig = sc.pl.dotplot(
        adata,
        var_names=var_names,
        groupby=groupby,
        swap_axes=False,
        standard_scale=standard_scale,
        figsize=figsize,
        return_fig=True,
        dendrogram=dendrogram,  # Adjusted to be within the valid range
        **kwargs,
    )

    # Customize the plot
    fig = (
        fig.add_totals()
        .style(largest_dot=largest_dot, cmap=cmap, grid=grid)
        .legend(width=legend_width)
    )

    # Save the plot to a file
    fig.savefig(save_path)
