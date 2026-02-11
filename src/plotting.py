"""
plotting.py

Contains all plots. 
"""


import os
import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm, colors
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np
import seaborn as sns
import networkx as nx


# Plot city boundary, roads, railways, and water features
def plot_base_map(boundary, roads, railways, water_polygons, water_edges, city_name = None, title=None, save_path=None):
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Water polygons
    if water_polygons is not None and not water_polygons.empty:
        water_polygons.plot(
            ax=ax,
            color="skyblue",
            edgecolor="none",
            alpha=0.4,
        )

    # Water edges (water polygons outline)
    if water_edges is not None and not water_edges.empty:
        water_edges.plot(
            ax=ax,
            color="navy",
            linewidth=0.4,
            alpha=0.6,
        )

    # Boundary
    if boundary is not None and not boundary.empty:
        boundary.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=1,
            linestyle="--",
        )

    # Roads
    if roads is not None and not roads.empty:
        roads.plot(
            ax=ax, 
            color="dimgray", 
            linewidth=0.2)

    # Railways
    if railways is not None and not railways.empty:
        railways.plot(
            ax=ax, 
            color="firebrick", 
            linewidth=0.4)

    # Title
    if title is None:
        title = f"{city_name.capitalize()} base map"
    ax.set_title(title, fontsize=18)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor="skyblue", edgecolor="navy", alpha=0.5, label="Water"),
        Line2D([0], [0], color="dimgray", linewidth=0.8, label="Roads"),
        Line2D([0], [0], color="firebrick", linewidth=1, label="Railways"),
        mpatches.Patch(facecolor="none", edgecolor="black", linestyle="--", linewidth=1, label="City boundary")
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=16)

    # Formatting
    ax.axis("off")
    plt.tight_layout()

    # Save (if requested)
    if save_path:
        filename = f"{city_name.lower()}_base.pdf"
        save_path = os.path.join(save_path, filename)
        plt.savefig(save_path)
        plt.close(fig)
    else:
        return fig, ax
    
    return fig, ax



# Plot water polygons and their edges, with optional city boundary.
def plot_water_map(water_polygons, water_edges, boundary=None, city_name=None, title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Water polygons (filled)
    if water_polygons is not None and not water_polygons.empty:
        water_polygons.plot(
            ax=ax,
            color="skyblue",
            edgecolor="none",
            alpha=0.4
        )

    # Water edges (outlines)
    if water_edges is not None and not water_edges.empty:
        water_edges.plot(
            ax=ax,
            color="navy",
            linewidth=0.4,
            alpha=0.6
        )

    # Boundary
    if boundary is not None and not boundary.empty:
        boundary.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=1,
            linestyle="--"
        )

    # Title
    if title is None:
        title = f"{city_name.capitalize()} water features"
    ax.set_title(title, fontsize=18)

    # Legend
    legend_handles = [
        mpatches.Patch(color="skyblue", alpha=0.4, label="Water polygons"),
        mpatches.Patch(facecolor="none", edgecolor="navy", linewidth=1, alpha=0.6, label="Water edges")
    ]
    if boundary is not None:
        legend_handles.append(
            mpatches.Patch(facecolor="none", edgecolor="black", linestyle="--", linewidth=1, label="City boundary")
        )

    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=16)

    # Formatting
    ax.axis("off")
    plt.tight_layout()

    # Save (if requested)
    if save_path:
        filename = f"{city_name.lower()}_water.pdf"
        save_path = os.path.join(save_path, filename)
        plt.savefig(save_path)
        plt.close(fig)
    else:
        return fig, ax

    return fig, ax


# Plot blocks
def plot_blocks(blocks, city_name=None, title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot generated blocks
    if blocks is not None and not blocks.empty:
        blocks.plot(
            ax=ax,
            facecolor='orange',
            edgecolor='black',
            linewidth=0.2,
            alpha=0.6
        )

    # Title
    if title is None:
        title = f"Blocks"
    full_title = f"{city_name.capitalize()} {title}" if city_name else title
    ax.set_title(full_title, fontsize=18)

    # Legend
    legend_handles = [
        mpatches.Patch(facecolor='orange', edgecolor='black', alpha=0.4, label='Blocks'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', frameon=True, fontsize=16)

    # Formatting
    ax.axis("off")
    plt.tight_layout()

    # Save (if requested)
    if save_path:
        # Create a safe filename
        parts = []
        if city_name:
            parts.append(city_name.lower())
        # Sanitize title: lowercase, remove special chars, replace spaces with underscores
        safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower())
        parts.append(safe_title)
        parts.append(".pdf")

        filename = "_".join(parts)
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, ax

    return fig, ax


# Plot all blocks, highlighting suspicious ones in red
def plot_blocks_with_suspicious(blocks, suspicious=None, city_name=None, title=None, save_path=None):
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if blocks is None or blocks.empty:
        # Nothing to plot
        ax.set_axis_off()
        return fig, ax

    # Determine remaining vs suspicious
    if (suspicious is not None) and (not suspicious.empty):
        remaining = blocks.drop(suspicious.index, errors="ignore")
    else:
        remaining = blocks
        suspicious = None

    # Plot remaining blocks
    if not remaining.empty:
        remaining.plot(
            ax=ax,
            facecolor='lightgray',
            edgecolor='black',
            linewidth=0.2,
            alpha=0.4
        )

    # Plot suspicious blocks (in red)
    if suspicious is not None and not suspicious.empty:
        suspicious.plot(
            ax=ax,
            facecolor='red',
            edgecolor='black',
            linewidth=0.3,
            alpha=0.8
        )

    # Title
    if title is None:
        title = "Blocks after removal (red = removed)"
    full_title = f"{city_name.capitalize()} {title}" if city_name else title
    ax.set_title(full_title, fontsize=18)

        # Legend
    legend_handles = [
        mpatches.Patch(facecolor='lightgray', edgecolor='black',
                       alpha=0.6, label='Blocks')
    ]
    if suspicious is not None and not suspicious.empty:
        legend_handles.append(
            mpatches.Patch(facecolor='red', edgecolor='black',
                           alpha=0.8, label='False water blocks')
        )

    ax.legend(handles=legend_handles, loc='upper right',
              frameon=True, fontsize=16)

    # Formatting
    ax.axis("off")
    plt.tight_layout()

    # Save (if requested)
    if save_path:
        parts = []
        if city_name:
            parts.append(city_name.lower())
        safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower())
        parts.append(safe_title)
        parts.append(".pdf")

        filename = "_".join(parts)
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, ax

    return fig, ax
    

# Plot blocks colored by POI count (log scale)
def plot_blocks_with_pois(blocks, city_name=None, title=None, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 8))

    # Handle empty data
    if blocks is not None and not blocks.empty:
        # Compute vmin/vmax for LogNorm (avoid log(0))
        data = blocks["poi_count"]
        vmin = max(1, data.min())   # at least 1
        vmax = data.max()

        blocks.plot(
            ax=ax,
            column="poi_count",
            cmap="OrRd",
            norm=colors.LogNorm(vmin=vmin, vmax=vmax),
            legend=True,
            legend_kwds={
                "label": "Number of POIs",
                "shrink": 0.6
            },
            edgecolor="white",
            linewidth=0.1
        )

        # Formatting and changing font sizes
        cbar = ax.get_figure().axes[-1]
        cbar.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        cbar.tick_params(labelsize=12)
        cbar.set_ylabel("Number of POIs", fontsize=14)

    # Title
    if title is None:
        title = "Blocks with POIs"
    full_title = f"{city_name.capitalize()} {title}" if city_name else title
    fig.suptitle(full_title, fontsize=16, y=0.95)

    # Formatting
    ax.axis("off")
    plt.tight_layout()

    # Save (if requested)
    if save_path:
        parts = []
        if city_name:
            parts.append(city_name.lower())
        # Sanitize title: lowercase, remove special chars, replace spaces with underscores
        safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower())
        parts.append(safe_title)
        parts.append(".pdf")

        filename = "_".join(parts)
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, ax

    return fig, ax

    
# Plot block adjacency graph colored/scaled by a chosen node attribute
def plot_block_graph(G, blocks, attr="poi_count", city_name=None, title=None, save_path=None):

    # Collect node values
    nodes = list(G.nodes())
    attr_values = np.array([G.nodes[n].get(attr, 0) for n in nodes], dtype=float)

    # Keep non-negative
    attr_values[attr_values < 0] = 0

    # Log scale setup
    if np.any(attr_values > 0):
        vmin = max(1, attr_values[attr_values > 0].min())
        vmax = attr_values.max()
    else:
        # all zeros – avoid log problems
        vmin, vmax = 1, 1

    # Values actually sent to LogNorm (no zeros)
    attr_for_color = np.where(attr_values <= 0, vmin, attr_values)

    norm = colors.LogNorm(vmin=vmin, vmax=vmax)
    cmap = cm.OrRd
    node_colors = [cmap(norm(v)) for v in attr_for_color]

    # Node sizes
    base_sizes = 3 + attr_values * 2
    node_sizes = np.clip(base_sizes, 5, 200)


    # Positions: block centroids
    pos = {
        row["block_id"]: (row.geometry.centroid.x, row.geometry.centroid.y)
        for _, row in blocks.iterrows()
        if row["block_id"] in G.nodes
    }

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="gray",
        alpha=0.2,
        width=0.5
    )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        alpha=0.9,
        linewidths=0.05,
        edgecolors="lightgray",
    )

    # Colorbar
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax, shrink=0.6)

    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel("Number of POIs", fontsize=14)


    # Title
    if title is None:
        title = "network graph"
    full_title = f"{city_name.capitalize()} {title}" if city_name else title
    fig.suptitle(full_title, fontsize=16, y=0.95)

    # Formatting
    ax.axis("off")
    ax.set_aspect("equal")
    plt.tight_layout()

    # Save (if requested)
    if save_path:
        parts = []
        if city_name:
            parts.append(city_name.lower())
        safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower())
        parts.append(safe_title)
        parts.append(".pdf")

        filename = "_".join(parts)
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, ax

    return fig, ax


# Plot largest component
def plot_largest_component(blocks, G, city_name=None, title=None, save_path=None):

    # Compute largest connected component 
    largest = max(nx.connected_components(G), key=len)
    blocks = blocks.copy()
    blocks["in_largest"] = blocks["block_id"].isin(largest)

    # Figure 
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot blocks
    blocks.plot(
        ax=ax,
        color=blocks["in_largest"].map({True: "darkorange", False: "lightgray"}),
        edgecolor="white",
        linewidth=0.1,
    )

    # Title
    if title is None:
        title = "Largest connected component"
    full_title = f"{city_name.capitalize()} {title}" if city_name else title
    fig.suptitle(full_title, fontsize=16, y=0.95)

    # Legend (match style)
    legend_handles = [
        mpatches.Patch(facecolor="darkorange", edgecolor="black", label="Largest connected component"),
        mpatches.Patch(facecolor="lightgray", edgecolor="black", label="Other components")
    ]

    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=16)

    # Formatting
    ax.axis("off")
    plt.tight_layout()

    # Save (if requested)
    if save_path:
        # Create a safe filename
        parts = []
        if city_name:
            parts.append(city_name.lower())
        # Sanitize title: lowercase, remove special chars, replace spaces with underscores
        safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", title.strip().lower())
        parts.append(safe_title)
        parts.append(".pdf")

        filename = "_".join(parts)
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, ax

    return fig, ax

def plot_ge_heatmap(df_ge, city_name=None, title="GE distances", order=None, pretty_labels=None, vmin=None, vmax=None, cmap=cm.OrRd, cbar_shrink=0.6, save_path=None):
    # Ordering
    if order is not None:
        df_plot = df_ge.loc[order, order]
    else:
        df_plot = df_ge.copy()
        order = list(df_plot.index)

    if pretty_labels is None:
        pretty_labels = order

    # Color scaling
    if vmin is None:
        vmin = df_plot.values.min()
    if vmax is None:
        vmax = df_plot.values.max()

    # Figure
    fig, ax = plt.subplots(figsize=(10, 10))

    hm = sns.heatmap(
        df_plot,
        annot=True,
        fmt=".3f",
        annot_kws={"fontsize": 14},
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        square=True,
        linewidths=0.4,
        linecolor="white",
        cbar=False,
        ax=ax,
    )

    # Colorbar
    sm = cm.ScalarMappable(norm=colors.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=cbar_shrink)
    cbar.ax.tick_params(labelsize=12)
    cbar.ax.set_ylabel("GE distance", fontsize=14, labelpad=10)

    # Labels & formatting
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticklabels(pretty_labels, rotation=45, ha="right", fontsize=16)
    ax.set_yticklabels(pretty_labels, rotation=0, fontsize=16)

    plt.tight_layout()

    if save_path:
        full_title = f"{city_name.capitalize()} {title}" if city_name else title
        safe_title = re.sub(r"[^a-zA-Z0-9]+", "_", full_title.lower())
        filename = f"{safe_title}.pdf"
        filepath = os.path.join(save_path, filename)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        return fig, ax

    return fig, ax


# Plot distribution of randomized variances vs observed variance for a given category
def plot_variance_distribution_from_results(category, df_z, label_map, CITY_NAME, save_path=None):

    rand_vars = df_z.loc[category, "rand_vars"]
    real_var  = df_z.loc[category, "real_var"]
    mean_rand = df_z.loc[category, "mean_rand"]
    std_rand  = df_z.loc[category, "std_rand"]
    z         = df_z.loc[category, "z_score"]

    pretty = label_map.get(category, category)

    # KDE peak calc (kept from your code)
    plt.figure(figsize=(8, 5))
    tmp = sns.kdeplot(rand_vars, bw_adjust=1, color="black")
    plt.clf()

    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(rand_vars, bins=30, stat='density',
                 color="lightgray", edgecolor=None, alpha=0.6, ax=ax)

    sns.kdeplot(rand_vars, bw_adjust=1, color="dimgray",
                linewidth=2, alpha=0.6, label="Distribution", ax=ax)

    ax.axvline(real_var, color="red", linestyle="--", linewidth=2,
               label=f"Observed variance = {real_var:.2f}")

    ax.axvline(mean_rand - std_rand, color="gray", linestyle="--", linewidth=1)
    ax.axvline(mean_rand + std_rand, color="gray", linestyle="--", linewidth=1)

    ax.set_xlabel("Variance")
    ax.set_ylabel("Density")
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, color="lightgray", alpha=0.3)

    # Title with city name
    ax.set_title(f"{CITY_NAME} - {pretty}: z-score = {z:.2f}", fontsize=16)

    ax.legend()
    fig.tight_layout()

    # Save if requested
    if save_path:

        filename = f"{CITY_NAME}_{category}.pdf"
        filepath = os.path.join(save_path, filename)

        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

    return fig, ax



# Plot POI distribution on a base
def plot_poi_distribution(boundary, roads, pois, city_name=None, title=None, save_path=None):

    # Figure / axis
    fig, ax = plt.subplots(figsize=(10, 7))

    # Boundary
    if boundary is not None and not boundary.empty:
        boundary.plot(
            ax=ax,
            facecolor="none",
            edgecolor="black",
            linewidth=1,
            linestyle="--",
        )

    # Roads
    if roads is not None and not roads.empty:
        roads.plot(
            ax=ax,
            color="dimgray",
            linewidth=0.2
        )

    # POIs (red dots, on top)
    if pois is not None and not pois.empty:
        pois.plot(
            ax=ax,
            markersize=3,
            color="red",
            alpha=0.6,
            zorder=5,
        )

    # Title
    if title is None and city_name is not None:
        title = f"{city_name.capitalize()} – POI distribution"
    elif title is None:
        title = "POI distribution"

    ax.set_title(title, fontsize=18)

    # Legend
    legend_handles = []

    # City boundary
    legend_handles.append(
        mpatches.Patch(
            facecolor="none",
            edgecolor="black",
            linestyle="--",
            linewidth=1,
            label="City boundary",
        )
    )

    # Roads
    legend_handles.append(
        Line2D(
            [0], [0],
            color="dimgray",
            linewidth=0.8,
            label="Roads",
        )
    )

    # POIs
    legend_handles.append(
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            color="red",
            markersize=6,
            label="POIs",
            alpha=0.8,
        )
    )

    ax.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=14)

    # Formatting
    ax.axis("off")
    plt.tight_layout()

    # Save (if requested)
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        if city_name is not None:
            filename = f"{city_name.lower()}_poi_distribution.pdf"
        else:
            filename = "poi_distribution.pdf"

        full_path = os.path.join(save_path, filename)
        plt.savefig(full_path)
        plt.close(fig)
    else:
        return fig, ax

    return fig, ax
