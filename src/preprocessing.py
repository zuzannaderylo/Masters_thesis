"""
preprocessing.py

Preprocessing of raw OSM data. Includes boundary filtering, 
geometry cleaning and preparation of layers for further analysis.
"""


import geopandas as gpd
import pandas as pd

# Method for exploding multilines into lines, but with keeping attributes
# Used so that geometries are in LineStrings (easier to work with)
def explode_multilines_with_attrs(gdf):

    # Create an empty list to store the new rows
    exploded_rows = []

    # Iterate over all rows
    for idx, row in gdf.iterrows():
        geom = row.geometry

        # Skip rows with missing geometry
        if geom is None:
            continue

        # If geometry is a MultiLineString, split it into its individual parts
        if geom.geom_type == "MultiLineString":
            for part in geom.geoms:
                new_row = row.copy()
                new_row.geometry = part
                exploded_rows.append(new_row)

        # If geometry is already a LineString, just keep it as it is
        elif geom.geom_type == "LineString":
            exploded_rows.append(row)

    return gpd.GeoDataFrame(exploded_rows, crs=gdf.crs)

### Roads
# Extract and filter roads to be used for block construction
def load_roads(osm):

    # Extract all driving roads
    driving_roads = osm.get_network(network_type="driving")

    # Explode MultiLineStrings into LineStrings
    driving_roads = explode_multilines_with_attrs(driving_roads)

    # Define road types that form urban blocks
    selected_types = [
        "motorway", "motorway_link", "trunk", "trunk_link",
        "primary", "primary_link", "secondary", "secondary_link",
        "tertiary", "tertiary_link", "unclassified", "residential",
        "living_street", "service", "pedestrian", "cycleway", "path"
    ]

    # Use .loc[] to filter 
    selected_roads = driving_roads.loc[
        driving_roads["highway"].isin(selected_types)
    ].copy()

    # Extract all ways tagged as bridge
    bridges = osm.get_data_by_custom_criteria(
        custom_filter={
            "bridge": ["yes"],
            "man_made": ["bridge"]
        },
        filter_type="keep",
        keep_nodes=False,
        keep_relations=True
    )

    # --- CLEANUP STEP ---
    if bridges is not None and not bridges.empty:
        # Drop anything that's not a line geometry
        bridges = bridges[bridges.geometry.type.isin(["LineString", "MultiLineString"])].copy()

        # Drop invalid geometries
        bridges = bridges[bridges.is_valid]

        # Explode MultiLineStrings into LineStrings
        bridges = bridges.explode(index_parts=False).reset_index(drop=True)

        # Ensure same CRS
        bridges = bridges.to_crs(selected_roads.crs)

        # Merge safely
        roads = pd.concat([selected_roads, bridges], ignore_index=True)
    else:
        roads = selected_roads

    return roads

### Railways
# Extract and filter railways to be used for block construction
def load_railways(osm):
   # Extract railway features (only main rail lines)
    railways = osm.get_data_by_custom_criteria(
        custom_filter={"railway": ["rail"]},
        filter_type="keep"
    )

    # Explode MultiLineStrings into LineStrings
    railways = explode_multilines_with_attrs(railways)

    return railways


### Water
# Load water-related polygons from OSM
def load_water_polygons(osm):
    water_polygons = osm.get_data_by_custom_criteria(
        custom_filter={
            "natural": ["water", "coastline", "bay", "wetland"],
            "landuse": ["reservoir", "basin"],
            "waterway": ["riverbank", "canal"],
        },
        filter_type="keep",
    )
    return water_polygons

# Prepare water geometries
def prepare_water_geometries(water_polygons):

    # Choose geometry types - polygons and multipolygons
    polygons = water_polygons[water_polygons.geometry.type.isin(["Polygon", "MultiPolygon"])].copy()

    water_polygons = polygons

    return water_polygons

# Convert water polygons to edge lines
def build_water_edges(water_polygons):
    
    # Build edges for block construction
    def to_edges(geom):
        if geom is None or geom.is_empty:
            return None
        if geom.geom_type in ["Polygon", "MultiPolygon"]:
            return geom.boundary
        elif geom.geom_type in ["LineString", "MultiLineString"]:
            return geom
        # Skip points, multipoints, etc.
        else:
            return None

    water_edges = water_polygons.copy()
    water_edges["geometry"] = water_edges["geometry"].apply(to_edges)

    # Drop invalid/empty
    water_edges = water_edges[~water_edges.geometry.is_empty]
    
    # Explode multilines into single lines
    water_edges = explode_multilines_with_attrs(water_edges)
    
    return water_edges


### Handling crs
# Find the best UTM projection for a GeoDataFrame
def get_local_crs(gdf):
    lon, lat = gdf.total_bounds[0::2].mean(), gdf.total_bounds[1::2].mean()
    zone = int((lon + 180) / 6) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return epsg

# Reproject all GeoDataFrames in a dict to the same EPSG
def reproject_all(layers: dict, epsg: int):
    for name, gdf in layers.items():
        if gdf is not None and hasattr(gdf, "crs"):
            layers[name] = gdf.to_crs(epsg=epsg)
    return layers

# Change data crs
def change_crs(boundary, roads, railways, water_polygons, pois):
    local_epsg = get_local_crs(boundary)
    print(f"Using local CRS: EPSG:{local_epsg}")
    
    # Reproject all at once
    layers = {
        "boundary": boundary,
        "roads": roads,
        "railways": railways,
        "water_polygons": water_polygons,
        "pois": pois
    }
    layers = reproject_all(layers, local_epsg)

    # Unpack
    boundary = layers["boundary"]
    roads = layers["roads"]
    railways = layers["railways"]
    water_polygons = layers["water_polygons"]
    pois = layers["pois"]

    return boundary, roads, railways, water_polygons, pois
