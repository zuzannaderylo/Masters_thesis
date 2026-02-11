"""
running_pipeline.py

The full analysis pipeline.
Sequentially executes data retrieval, preprocessing, block construction,
graph creation, network analysis and livability score computation.
The main code in this thesis.
"""


import os
import numpy as np
import pandas as pd

from src.data_retrieval import init_osm
from src.preprocessing import (
    load_roads, load_railways, load_water_polygons,
    get_local_crs, reproject_all,
    build_water_edges, prepare_water_geometries
)
from src.prepare_pois import load_pois_with_green, prepare_pois, assign_pois_to_blocks
from src.process_blocks import (
    construct_blocks, filter_water_blocks, filter_small_blocks,
    filter_irregular_blocks, remove_false_water_blocks
)
from src.plotting import (
    plot_base_map, plot_block_graph, plot_blocks, plot_blocks_with_pois, plot_blocks_with_suspicious, plot_ge_heatmap,
    plot_largest_component, plot_variance_distribution_from_results,
    plot_poi_distribution, plot_water_map
)
from src.construct_graph import build_block_graph
from src.network_analysis import (
    precompute_resistance, get_largest_component,
    prepare_category_dicts, compute_generalized_euclidean_matrix,
    load_Q, compute_and_store_Q, compute_variance, compute_z_scores
)
from src.network_distance import ge
from src.livability_scores import (
    add_inv_degree_weights, compute_livability_from_matrix,
    compute_livability_normalized_1, normalize_ge_rowwise, normalize_ge_sum_to_one
)


# The whole pipeline code - with city name as parameter
def run_pipeline(CITY_NAME):
    
    print(f"\n=== Starting pipeline for: {CITY_NAME} ===")

    # ----- 0. Create output folders -----
    # Create main and subfolders if they don't exist
    os.makedirs(CITY_NAME, exist_ok=True)
    save_dir = CITY_NAME

    os.makedirs(os.path.join(save_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "data"), exist_ok=True)

    data_save_path = os.path.join(save_dir, "data")
    figures_path = os.path.join(save_dir, "figures")

    print(f"Output folders ready: {figures_path} and {data_save_path}")

    # For plotting z-scores
    order = [
        "food",
        "retail",
        "education",
        "healthcare",
        "infrastructure_transport",
        "culture_leisure",
        "green_spaces",
        "public_services",
        "other_daily_utilities",
    ]
    pretty_labels = [
        "Food",
        "Retail",
        "Education",
        "Healthcare",
        "Infrastructure & transport",
        "Culture & leisure",
        "Green spaces",
        "Public services",
        "Other daily utilities",
    ]
    label_map = dict(zip(order, pretty_labels))

    # ----- 1. Data retrieval (osm init + boundary) -----
    print("Initializing OSM...")
    osm, boundary = init_osm(CITY_NAME, data_save_path)

    # ----- 2. Preprocessing -----
    # Load data
    print("Loading raw layers from OSM (roads, railways, water, POIs)...")
    roads = load_roads(osm)
    railways = load_railways(osm)
    water_polygons = load_water_polygons(osm)
    pois = load_pois_with_green(osm)

    # Check 
    print(f"Loaded {len(roads):,} roads")
    print(f"Loaded {len(railways):,} railways")
    print(f"Loaded {len(water_polygons):,} water polygons")

    plot_poi_distribution(boundary, roads, pois, CITY_NAME, "POI distribution", save_path=figures_path)

    # Change crs. Compute city-local UTM projection
    print("Computing city-local CRS and reprojecting all layers...")
    local_epsg = get_local_crs(boundary)
    print(f"Using local CRS: EPSG:{local_epsg}")

    # Reproject all layers at once
    layers = {
        "boundary": boundary,
        "roads": roads,
        "railways": railways,
        "water_polygons": water_polygons,
        "pois": pois
    }
    layers = reproject_all(layers, local_epsg)

    # Unpack the reprojected layers
    boundary = layers["boundary"]
    roads = layers["roads"]
    railways = layers["railways"]
    water_polygons = layers["water_polygons"]
    pois = layers["pois"]

    # Build water edges after reprojection (they are used for constructing blocks)
    print("Building water edges from water polygons...")
    water_edges = build_water_edges(layers["water_polygons"])

    # Change water geometries
    print("Preparing water geometries...")
    water_polygons = prepare_water_geometries(water_polygons)

    # Prepare pois
    print("Preparing POIs...")
    pois = prepare_pois(pois=pois)

    plot_base_map(boundary, roads, railways, water_polygons, water_edges, city_name=CITY_NAME, save_path=figures_path)
    plot_water_map(water_polygons, water_edges, boundary = None, city_name=CITY_NAME, save_path=figures_path)

    # ----- 3. Block construction and filtering -----

    # Create initial blocks
    print("Constructing initial blocks...")
    initial_blocks = construct_blocks(roads, railways, water_edges)

    # Process blocks
    print("Filtering blocks...")
    blocks_no_water = filter_water_blocks(initial_blocks, water_polygons)
    blocks_no_small = filter_small_blocks(blocks_no_water)
    blocks_cleaned, suspicious = remove_false_water_blocks(blocks_no_small, area_quantile=0.999, compactness_quantile=0.03)
    blocks_no_irregular = filter_irregular_blocks(blocks_cleaned)

    blocks = blocks_no_irregular

    plot_blocks(blocks=initial_blocks, city_name=CITY_NAME, title="initial blocks", save_path=figures_path)
    plot_blocks(blocks=blocks_no_water, city_name=CITY_NAME, title="blocks after water filtering", save_path=figures_path)
    plot_blocks(blocks=blocks_no_small, city_name=CITY_NAME, title="blocks after small filtering", save_path=figures_path)
    plot_blocks_with_suspicious(blocks=blocks_cleaned, suspicious=suspicious, city_name=CITY_NAME, title="false water blocks", save_path=figures_path)
    plot_blocks(blocks=blocks_no_irregular, city_name=CITY_NAME, title="blocks after irregular filtering", save_path=figures_path)
    plot_blocks(blocks=blocks_no_irregular, city_name=CITY_NAME, title="final blocks", save_path=figures_path)

    # Check block counts
    print(f"Initial blocks constructed: {len(initial_blocks):,}")
    print(f"Blocks after removing water blocks: {len(blocks_no_water):,}")
    print(f"Blocks after merging small blocks: {len(blocks_no_small):,}")
    print(f"Blocks after removing false water blocks: {len(blocks_cleaned):,}")
    print(f"Final number of blocks: {len(blocks):,}")

    print("Final block layer prepared")


    # ----- 4. Assign POIs to blocks -----
    print("Assigning POIs to blocks...")
    blocks_with_pois = assign_pois_to_blocks(pois, blocks)
    blocks = blocks_with_pois

    plot_blocks_with_pois(blocks=blocks_with_pois, city_name=CITY_NAME, title="block-level POI density", save_path=figures_path)


    # ----- 5. Graph construction -----
    # Build graph from blocks with POIs
    print("Building block adjacency graph...")
    G = build_block_graph(blocks)

    plot_block_graph(G, blocks, city_name=CITY_NAME, save_path=figures_path)

    # Select largest connected component
    print("Extracting largest connected component...")
    G_largest = get_largest_component(G)

    plot_largest_component(blocks, G, CITY_NAME, title="largest connected component", save_path=figures_path)


    # ----- 6. Network analysis (Q, GE, resistance, variance, z-scores) -----

    # Compute Q
    print("Computing and storing Q matrix for largest component...")
    Q = compute_and_store_Q(G_largest, CITY_NAME, data_save_path)

    # Load 
    G_largest, Q = load_Q(CITY_NAME, data_save_path)

    # Prepare category dictionaries
    print("Preparing category dictionaries for nodes in largest component...")
    category_counts = blocks.attrs["category_counts"]
    category_dicts_largest = prepare_category_dicts(category_counts, G_largest)

    # Compute GE matrix
    print("Computing generalized Euclidean (GE) matrix...")
    df_ge = compute_generalized_euclidean_matrix(G_largest, category_dicts_largest, ge=ge, Q_func=lambda G: Q)
    df_ge.to_csv(os.path.join(data_save_path, f"{CITY_NAME}_results_GE.csv"), index=False)

    print(f"GE matrix size: {df_ge.shape[0]} × {df_ge.shape[1]}")
    print(f"Number of categories: {len(category_dicts_largest)}")
    
    # Plot heatmap
    plot_ge_heatmap(df_ge, city_name=CITY_NAME, order=order, pretty_labels=pretty_labels, save_path=figures_path)

    # Precompute resistance once
    print("Precomputing resistance matrix...")
    resistance_matrix = precompute_resistance(G_largest)

    # Compute variance for each category (using precomputed resistance matrix)
    print("Computing variance per category...")
    variance_categories = compute_variance(category_dicts_largest, G_largest, resistance_matrix)

    # Compute z-scores
    print("Computing z-scores...")
    df_z = compute_z_scores(category_dicts_largest, G_largest, resistance_matrix, n_iter=1000)

    # Store after computing df_z
    df_z.to_csv(os.path.join(data_save_path, f"{CITY_NAME}_results_z_scores.csv"), index=True)

    # Plotting z-scores for each category
    for category in df_z.index:
        plot_variance_distribution_from_results(category, df_z, label_map, CITY_NAME, save_path=figures_path)


    # ----- 7. Livability score -----
    # Align df_z to the same order as df_ge. So they correspond exactly
    df_z_aligned = df_z.reindex(df_ge.index)

    # Save z-scores
    z_scores = df_z_aligned["z_score"].values

    # Livabiity score for the city
    print("Computing livability score (unnormalized)...")
    livability_unnormalized = compute_livability_from_matrix(df_ge, z_scores)
    print("Livability score:", livability_unnormalized)


    # ----- 8. Normalization variants -----
    
    # Normalization 1 - use the average instead of the sum
    print("Computing livability score variant: average instead of sum...")
    livability_normalized1 = compute_livability_normalized_1(df_ge, z_scores)
    print("Livability score (with using average instead of sum):", livability_normalized1)

    # Normalization 2 - add weights to the edges in graph
    print("Computing livability score variant: weighted graph...")

    # Make a weighted copy of the graph
    G_weighted = G_largest.copy()

    # Add degree-based weights
    G_weighted = add_inv_degree_weights(G_weighted)

    # Recompute Q and GE matrix using the weighted graph
    Q_weighted = compute_and_store_Q(G_weighted, CITY_NAME, output_dir=data_save_path)

    # Prepare category dictionaries
    category_dicts_largest_weighted = prepare_category_dicts(category_counts, G_weighted)

    # Compute GE matrix
    df_ge_weighted = compute_generalized_euclidean_matrix(
        G_weighted, category_dicts_largest_weighted, ge=ge, Q_func=lambda G: Q_weighted
    )

    resistance_matrix_weighted = precompute_resistance(G_weighted)

    # Compute variance for each category (using precomputed resistance matrix)
    variance_categories_weighted = compute_variance(category_dicts_largest_weighted, G_weighted, resistance_matrix_weighted)

    # Compute z-scores
    df_z_weighted = compute_z_scores(category_dicts_largest_weighted, G_weighted, resistance_matrix_weighted, n_iter=1000)

    # Align df_z to the same order as df_ge. So they correspond exactly
    df_z_aligned_weighted = df_z_weighted.reindex(df_ge_weighted.index)

    # Save z-scores
    z_scores_weighted = df_z_aligned_weighted["z_score"].values

    # Compute livability score
    livability_normalized2 = compute_livability_from_matrix(df_ge_weighted, z_scores_weighted)

    print("Livability with weighted edges:", livability_normalized2)


    # Normalization 3 - normalizing generalized euclidean

    # Normalize GE matrix so that all values sum to 1 - global 
    print("Computing livability score variant: GE normalized to sum=1...")
    df_ge_sum1 = normalize_ge_sum_to_one(df_ge)

    # Compute livability again using normalized GE
    livability_normalized3 = compute_livability_from_matrix(df_ge_sum1, z_scores)
    print("Livability score (with normalized GE):", livability_normalized3)

    # Row normalization; per category. Every row sum up to 1
    print("Computing livability score variant: GE row-normalized (each row sums to 1)...")
    df_ge_row = normalize_ge_rowwise(df_ge)

    # Compute livability again using row-normalized GE
    livability_rowwise = compute_livability_from_matrix(df_ge_row, z_scores)
    print("Livability score (row-normalized GE):", livability_rowwise)


    # Normalization 4 - Logarithm of ge
    print("Computing livability score variant: log(GE)...")

    # Copy GE matrix
    GE = df_ge.values.astype(float)

    # Replace zeros to avoid log(0)
    GE[GE <= 0] = 1e-9

    # Apply logarithm
    GE_log = np.log(GE)

    # Create DataFrame again
    df_ge_log = pd.DataFrame(GE_log, index=df_ge.index, columns=df_ge.columns)

    # Compute livability again
    livability_log = compute_livability_from_matrix(df_ge_log, z_scores)
    print("Livability (log-transformed GE):", livability_log)

    
    # ----- 9. Final score -----
    final_score = livability_unnormalized

    print(f"\nFinal livability score for {CITY_NAME} is {final_score:.4f}.")
    print(f"=== Pipeline finished for: {CITY_NAME} ===\n")

    return final_score