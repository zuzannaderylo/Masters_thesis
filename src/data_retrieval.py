"""
data_retrieval.py

Retrieval of raw OSM data for a city.
Includes loading of administrative boundaries, street networks,
railways, water features and POIs.
"""


from pyrosm import OSM, get_data
from shapely.geometry import Polygon, MultiPolygon

# Mapping from city name to (OSM boundary name, admin_level)
# https://wiki.openstreetmap.org/wiki/Template:Admin_level
CITY_CONFIG = {
    "Copenhagen": {"boundary_name": "Københavns Kommune", "admin_level": 7},
    "Gdansk": {"boundary_name": "Gdańsk", "admin_level": 8},
    # Add more cities here
}

# Remove interior holes from polygons
def drop_holes(geom):
    if geom.geom_type == "Polygon":
        return Polygon(geom.exterior)
    elif geom.geom_type == "MultiPolygon":
        return MultiPolygon([Polygon(p.exterior) for p in geom.geoms])
    else:
        return geom
    

# Initialize OSM object for a city, clipped to its administrative boundary
# Update set to false (for the thesis), but for more up-to-date results change to true
def init_osm(city_name, directory="data"):

    # Download OSM data (.pbf file) for the given city
    fp = get_data(city_name, directory=directory, update=False)

    # Initialize OSM object
    osm = OSM(fp)

    # Look up boundary name and admin level in CITY_CONFIG
    cfg = CITY_CONFIG[city_name]
    boundary_name = cfg["boundary_name"]
    admin_level = cfg["admin_level"]

    # Load only boundaries with that name
    boundary = osm.get_boundaries(name=boundary_name)
    boundary = boundary[boundary["name"] == boundary_name]

    # If admin_level is specified, filter further
    if admin_level is not None:
        boundary = boundary[boundary["admin_level"] == str(admin_level)]

    if boundary.empty:
        raise ValueError(f"No boundary found for {city_name} with admin_level={admin_level}")

    # Clean geometry (keep only outer shells, without any holes)
    boundary["geometry"] = boundary["geometry"].apply(drop_holes)

    # Keep only the largest polygon
    boundary["geometry"] = boundary["geometry"].apply(
        lambda g: max(g.geoms, key=lambda p: p.area) if g.geom_type == "MultiPolygon" else g
    )

    # Extract geometry of the boundary
    bbox_geom = boundary['geometry'].values[0]

    # Re-initialize OSM with bounding box
    osm = OSM(fp, bounding_box=bbox_geom)

    return osm, boundary