"""
prepare_pois.py

Cleans, filters and categorizes Points of Interest (POIs).
Converts geometries to point representations and assigns
categories used in the analysis.
"""


import pandas as pd
import geopandas as gpd

# Defining mapping from "amenity" column to categories
amenity_to_category = {
    # Food
    "bar": "food", 
    "restaurant": "food", 
    "cafe": "food", 
    "ice_cream": "food",
    "fast_food": "food", 
    "pub": "food", 
    "hookah_lounge": "food",
    "food_court": "food", 
    "internet_cafe": "food", 
    "food_sharing": "food",
    "pastry": "food", 
    "community_centre;cafe": "food", 
    "canteen": "food",
    "biergarten": "food",

    # Infrastructure & transport
    "parking": "infrastructure_transport", 
    "parking_space": "infrastructure_transport",
    "bicycle_parking": "infrastructure_transport", 
    "motorcycle_parking": "infrastructure_transport",
    "charging_station": "infrastructure_transport", 
    "taxi": "infrastructure_transport",
    "ferry_terminal": "infrastructure_transport", 
    "car_rental": "infrastructure_transport",
    "car_wash": "infrastructure_transport", 
    "bicycle_rental": "infrastructure_transport",
    "bus_station": "infrastructure_transport", 
    "car_sharing": "infrastructure_transport",
    "scooter_parking": "infrastructure_transport",
    "traffic_park": "infrastructure_transport",
    "motorcycle_rental": "infrastructure_transport",
    "kick-scooter_parking": "infrastructure_transport",

    # Education
    "school": "education", 
    "kindergarten": "education", 
    "childcare": "education",
    "university": "education",
    "college": "education", 
    "language_school": "education",
    "research_institute": "education", 
    "music_school": "education",
    "prep_school": "education",

    # Culture & leisure
    "social_facility": "culture_leisure", 
    "events_venue": "culture_leisure",
    "theatre": "culture_leisure", 
    "library": "culture_leisure", 
    "cinema": "culture_leisure",
    "gambling": "culture_leisure", 
    "music_venue": "culture_leisure",
    "arts_centre": "culture_leisure", 
    "casino": "culture_leisure",
    "nightclub": "culture_leisure", 
    "stripclub": "culture_leisure", 
    "brothel": "culture_leisure",
    "gallery": "culture_leisure", 
    "swingerclub": "culture_leisure",
    "monastery": "culture_leisure",
    "dojo": "culture_leisure", 
    "dive_centre": "culture_leisure",
    "exhibition_centre": "culture_leisure", 
    "planetarium": "culture_leisure",
    "public_bath": "culture_leisure", 
    "festival_grounds": "culture_leisure",
    "climbing_wall": "culture_leisure", 
    "dancing_school": "culture_leisure",
    "surf_school": "culture_leisure",
    "convent": "culture_leisure",

    # Public services
    "place_of_worship": "public_services", 
    "community_centre": "public_services",
    "bank": "public_services", 
    "post_office": "public_services",
    "police": "public_services", 
    "courthouse": "public_services",
    "fire_station": "public_services", 
    "social_centre": "public_services",
    "conference_centre": "public_services", 
    "funeral_hall": "public_services",
    "crematorium": "public_services", 
    "townhall": "public_services",
    "parliament": "public_services", 
    "lost_property_office": "public_services",
    "animal_shelter": "public_services",
    'local government unit': "public_services",
    "ranger_station": "public_services",

    # Healthcare
    "pharmacy": "healthcare", 
    "clinic": "healthcare", 
    "dentist": "healthcare",
    "doctors": "healthcare", 
    "veterinary": "healthcare", 
    "hospital": "healthcare",
    "nursing_home": "healthcare", 
    "fysioterapi": "healthcare",
    "healthcare": "healthcare",

    # Retail
    "marketplace": "retail",

    # Green spaces
    "playground": "green_spaces",

    # Other daily utilities
    "recycling": "other_daily_utilities", 
    "toilets": "other_daily_utilities",
    "drinking_water": "other_daily_utilities", 
    "atm": "other_daily_utilities",
    "fuel": "other_daily_utilities", 
    "parcel_locker": "other_daily_utilities",
    "bicycle_repair_station": "other_daily_utilities", 
    "coworking_space": "other_daily_utilities",
    "bureau_de_change": "other_daily_utilities", 
    "luggage_locker": "other_daily_utilities",
    "locker": "other_daily_utilities", 
    "left_luggage": "other_daily_utilities",
    "self_storage": "other_daily_utilities"
}

# Define mappings for other useful columns - for rows that didn't have "amenity" value
other_columns_to_category = {
    # Public services
    "office": "public_services",
    "post_office": "public_services",
    "charity": "public_services",
    "police": "public_services",

    # Culture & leisure
    "attraction": "culture_leisure",
    "camp_site": "culture_leisure",
    "information": "culture_leisure",
    "museum": "culture_leisure",
    "tourism": "culture_leisure",
    "caravan_site": "culture_leisure",
    "zoo": "culture_leisure",
    "swimming_pool ": "culture_leisure",

    # Food
    "bar": "food",
    "tea": "food",
    "pastry": "food",
    "restaurant ": "food",

    # Retail
    "books": "retail",
    "butcher": "retail",
    "clothes": "retail",
    "confectionery": "retail",
    "craft": "retail",
    "furniture": "retail",
    "gift": "retail",
    "massage": "retail",
    "model": "retail",
    "music": "retail",
    "outdoor": "retail",
    "pet": "retail",
    "second_hand": "retail",
    "wholesale": "retail",
    "shop": "retail",
    "shoes": "retail",
    "medical_supply ": "retail",
    
    # Infrastructure & transport
    "bicycle_rental": "infrastructure_transport",

    # Green spaces
    "green_spaces": "green_spaces"
}

# Extract raw POIs from OSM
def load_pois(osm):
    pois = osm.get_pois()
    return pois

# Process POIs: clean geometries and basic filtering
def process_pois(pois):

    # Drop empty or null geometries
    pois = pois[~pois.geometry.is_empty & pois.geometry.notna()].copy()

    # Drop invalid geometry types (multipolygons and multilinestrings)
    invalid_types = ['MultiPolygon', 'MultiLineString']
    pois = pois[~pois.geometry.type.isin(invalid_types)].copy()

    # Convert Polygon and LineString to centroid points
    pois.loc[pois.geometry.type.isin(['Polygon', 'LineString']), 'geometry'] = \
        pois.loc[pois.geometry.type.isin(['Polygon', 'LineString']), 'geometry'].centroid
    
    return pois


# Assign categories based on OSM tags
def assign_categories(pois):

    # Normalize - treat NaN, None and no as missing
    def normalize(val):
        if val is None:
            return None
        v = str(val).strip().lower()
        return None if v in {"nan", "none", "no", ""} else v

    # Check columns and assign category when a match is found
    def assign_category(row, ordered_mapping):
        for col, cat in ordered_mapping.items():
            if col in row and normalize(row[col]) is not None:
                return cat
        return None

    # Map directly from amenity_to_category
    pois["category"] = pois["amenity"].map(amenity_to_category)
    
    # Handle green spaces
    if "green_spaces" in pois.columns:
        pois.loc[pois["green_spaces"] == "yes", "category"] = "green_spaces"

    # For remaining, check other columns
    missing_mask = pois["category"].isna()
    pois_no_cat = pois.loc[missing_mask].copy()

    if not pois_no_cat.empty:
        pois_no_cat["category"] = pois_no_cat.apply(lambda r: assign_category(r, other_columns_to_category), axis=1)
        pois.loc[pois_no_cat.index, "category"] = pois_no_cat["category"]

    # Check 
    print(f"Assigned categories for {pois['category'].notna().sum()} POIs "
          f"({pois['category'].notna().mean():.1%} coverage)")
    
    return pois

# Method doing all
def prepare_pois(pois):
    pois = process_pois(pois)
    print("Number of POIs after cleaning: " + str(len(pois)))

    pois = assign_categories(pois)
    print("Number of POIs after categorization: " + str(len(pois)))

    return pois


# Green spaces POIS from the "leisure" tag
def load_pois_with_green(osm):

    # Load all regular POIs
    pois = load_pois(osm)
    print(f"Loaded {len(pois)} regular POIs")

    # Load green areas (parks, gardens, etc.)
    green_spaces = osm.get_data_by_custom_criteria(
        custom_filter={
            "leisure": ["park"],
            "landuse": ["recreation_ground"]
        },
        filter_type="keep"
    )

    if not green_spaces.empty:
        print(f"Loaded {len(green_spaces)} green-space features")

        # Filter out invalid geometries
        green_spaces = green_spaces[~green_spaces.geometry.is_empty & green_spaces.geometry.notna()].copy()

        # Add category
        green_spaces["green_spaces"] = "yes"

        # Merge
        pois = pd.concat([pois, green_spaces], ignore_index=True)

    print(f"Total combined POIs: {len(pois)}")
    return pois

# Assigning POIs to blocks
def assign_pois_to_blocks (pois, blocks):
    
    blocks = blocks.copy()
    pois = pois.copy()

    # Rename 'id' to 'poi_id'
    pois = pois.rename(columns={'id': 'poi_id'})

    # Make sure that there ar no rows with all none values
    pois = pois.dropna(how='all')

    # Keep only categorized POIs
    pois = pois[pois["category"].notna()].copy()

    # Spatial join (assign POIs to blocks)
    pois_with_blocks = gpd.sjoin(pois, blocks, how="left", predicate="within")

    # Count POIs by category per block
    category_counts = (
        pois_with_blocks.groupby(["block_id", "category"])
        .size()
        .reset_index(name="count")
    )

    # Group POI IDs per block
    pois_grouped = (
        pois_with_blocks.groupby("block_id")
        .agg({"poi_id": list})
        .reset_index()
    )

    # Merge POIs info into final_blocks
    blocks_with_pois  = blocks.merge(pois_grouped, on="block_id", how="left")

    # Create a new column that counts how many POIs each block has
    blocks_with_pois["poi_count"] = blocks_with_pois["poi_id"].apply(
        lambda x: len(x) if isinstance(x, list) else 0
    )

    # Store category_counts
    blocks_with_pois.attrs["category_counts"] = category_counts

    print(f"POIs assigned to {blocks_with_pois['poi_count'].gt(0).sum()} blocks "
          f"(out of {len(blocks_with_pois)})")

    return blocks_with_pois