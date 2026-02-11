"""
process_blocks.py

Constructs and filters urban blocks from spatial boundaries.
Includes polygonization of roads, railways and water features,
as well as removal or merging of water, small and irregular blocks.
"""


import geopandas as gpd
from matplotlib import pyplot as plt
import numpy as np
import shapely
from shapely.ops import polygonize, linemerge
import pandas as pd

# Create initial city blocks (polygons) from roads, railways and water edges
def construct_blocks(roads, railways, water_edges):
    # Merge boundaries into one GeoDataFrame
    boundaries = gpd.GeoDataFrame(
        pd.concat([roads, railways, water_edges], ignore_index=True),
        crs=roads.crs
    )

    # Create a single MultiLineString network
    merged_boundaries = linemerge(shapely.union_all(boundaries.geometry))
    
    # Polygonize into blocks
    blocks = list(polygonize(merged_boundaries))

    # Convert to GeoDataFrame
    blocks = gpd.GeoDataFrame(geometry=blocks, crs=boundaries.crs)

    return blocks

# Remove blocks dominated by water
def filter_water_blocks(blocks, water_polygons):
    # Add blocks IDs, to keep track of them
    blocks = blocks.reset_index(drop=True)
    blocks["block_id"] = blocks.index

    # Spatial join to find blocks that intersect with water. Add it as a flag column
    joined = gpd.sjoin(blocks, water_polygons, how="left", predicate="within")
    joined["is_water"] = ~joined["index_right"].isna()

    # Split blocks into water and non-water
    water_blocks = joined[joined["is_water"]].copy()
    cleaned_blocks = joined[~joined["is_water"]].copy()

    # Drop helper columns
    cleaned_blocks = cleaned_blocks.drop(columns=["index_right", "is_water"]).reset_index(drop=True)

    # Remove empty/broken geometries
    cleaned_blocks = cleaned_blocks[~cleaned_blocks.geometry.is_empty & cleaned_blocks.geometry.notnull()].reset_index(drop=True)

    return cleaned_blocks

# Filter out small blocks
def filter_small_blocks(blocks, lower_percentile=25, max_iterations=5):

    # Compute block areas ---
    blocks = blocks.reset_index(drop=True).copy()
    blocks["block_id"] = blocks.index

    # Compute current area and threshold
    blocks["area_m2"] = blocks.geometry.area
    min_area = blocks["area_m2"].quantile(lower_percentile / 100)

    for iteration in range(1, max_iterations + 1):

        # Compute current areas based on merged geometries
        blocks["area_m2"] = blocks.geometry.area
        small_blocks = blocks[blocks["area_m2"] < min_area].copy()
        valid_blocks = blocks[blocks["area_m2"] >= min_area].copy()      
        
        # Stop if there's no small blocks
        if small_blocks.empty:
            print("No small blocks left")
            break

        # Compute centroids for distance calculations
        valid_blocks["centroid"] = valid_blocks.geometry.centroid
        merged_ids = []

        # Merge small blocks into nearest valid blocks
        for idx, small_row in small_blocks.iterrows():
            small_centroid = small_row.geometry.centroid
            if valid_blocks.empty:
                continue

            distances = valid_blocks["centroid"].distance(small_centroid)
            nearest_idx = distances.idxmin()

            merged_geom = small_row.geometry.union(valid_blocks.loc[nearest_idx, "geometry"])
            valid_blocks.at[nearest_idx, "geometry"] = merged_geom
            merged_ids.append(small_row["block_id"])

        # Combine both sets before removing merged ones
        merged_blocks = pd.concat([valid_blocks, small_blocks], ignore_index=True)

        # Drop centroid helper column
        merged_blocks = merged_blocks.drop(columns=["centroid"], errors="ignore")

        # Remove the small blocks that were merged into others
        blocks = merged_blocks[~merged_blocks["block_id"].isin(merged_ids)].reset_index(drop=True)

        # Clean geometries and reassign IDs for next iteration
        blocks = blocks[~blocks.geometry.is_empty & blocks.geometry.notnull()].reset_index(drop=True)
        blocks["block_id"] = blocks.index
        blocks["area_m2"] = blocks.geometry.area

        print(f" Merged {len(merged_ids)} small blocks.")
        print(f" Remaining blocks = {len(blocks)}")

    # Final cleanup 
    blocks = blocks[~blocks.geometry.is_empty & blocks.geometry.notnull()].reset_index(drop=True)

    return blocks


# Filtering of irregular blocks
def filter_irregular_blocks(blocks, compactness_threshold=0.05, max_iterations=5):
    # Perimeter is length
    blocks["perimeter"] = blocks.geometry.length

    # Compactness: Polsby-Popper (4π * Area / Perimeter²)
    # Closer to 1 means more compact
    blocks["compactness"] = 4 * np.pi * blocks["area_m2"] / blocks["perimeter"]**2

    # Determine threshold
    threshold_value = blocks["compactness"].quantile(compactness_threshold)

    # Iterative merging
    for iteration in range (max_iterations):
        blocks["perimeter"] = blocks.geometry.length
        blocks["compactness"] = 4 * np.pi * blocks["area_m2"] / blocks["perimeter"] ** 2

        # Find irregular and compact blocks
        irregular = blocks[blocks["compactness"] <= threshold_value]
        compact = blocks[blocks["compactness"] > threshold_value]

        # Merge irregular blocks
        compact["centroid"] = compact.geometry.centroid
        merge_count = 0
        for idx, row in irregular.iterrows():
            centroid = row.geometry.centroid
            distances = compact["centroid"].distance(centroid)
            nearest_idx = distances.idxmin()
            merged_geom = row.geometry.union(compact.loc[nearest_idx, "geometry"])
            compact.at[nearest_idx, "geometry"] = merged_geom
            merge_count += 1

        compact = compact.drop(columns=["centroid"]).reset_index(drop=True)
        compact["area_m2"] = compact.geometry.area
        blocks = compact
        
    # Drop helper columns
    blocks = blocks.drop(columns=["perimeter", "compactness"], errors="ignore")
    
    return blocks

# Combine filtering steps
def filter_blocks(initial_blocks, water):
    # Filtering out water blocks
    blocks_no_water = filter_water_blocks(initial_blocks, water)

    # Filtering out small blocks
    blocks_no_small = filter_small_blocks(blocks_no_water)

    # Filtering out irregular blocks
    blocks_no_irregular = filter_irregular_blocks(blocks_no_small)

    # Blocks after filtering
    blocks = blocks_no_irregular

    return blocks


# Detect and remove large irregular blocks (e.g. false 'water' areas caused by boundary errors).
def remove_false_water_blocks(blocks, area_quantile=0.99, compactness_quantile=0.1, plot=True):
    blocks = blocks.copy()

    # Compute compactness (Polsby–Popper)
    blocks["compactness"] = 4 * np.pi * blocks.area / (blocks.length ** 2)

    # Define thresholds
    area_thr = blocks.area.quantile(area_quantile)
    comp_thr = blocks["compactness"].quantile(compactness_quantile)

    # Identify suspicious blocks (large + irregular)
    suspicios = blocks[(blocks.area > area_thr) & (blocks["compactness"] < comp_thr)]

    # Remove suspicious blocks
    cleaned = blocks.drop(suspicios.index)

    print(f"Removed {len(suspicios)} suspected false-water blocks "
          f"(area>{area_quantile*100:.0f}%, compactness<{compactness_quantile*100:.0f}%).")

    return cleaned, suspicios
