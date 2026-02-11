"""
construct_graph.py

Builds the block adjacency graph.
Each node represents an urban block and edges represent
spatial adjacency between blocks.
"""


import networkx as nx

# Build an undirected graph where each block is a node and edges connect touching blocks
def build_block_graph(blocks):
    blocks = blocks.copy()

    # Initialize graph
    G = nx.Graph()

    # Add each polygon as a node, using its block id as identifier
    for idx, row in blocks.iterrows():
        G.add_node(row["block_id"], geometry=row.geometry)

    # Use spatial index for efficient neighbor finding
    sindex = blocks.sindex    

    # For each block, find and connect neighboring blocks
    for idx, row in blocks.iterrows():
        geom = row.geometry
        block_id = row["block_id"]

        # Find possible neighbors based on bounding box
        possible_neighbors_index = list(sindex.intersection(geom.bounds))

        for neighbor_idx in possible_neighbors_index:
            neighbor_row = blocks.iloc[neighbor_idx]
            neighbor_geom = neighbor_row.geometry
            neighbor_id = neighbor_row["block_id"]

            # Avoid self-comparison
            if block_id == neighbor_id:
                continue

            # If geometries touch (share a boundary), add an edge
            if geom.touches(neighbor_geom):
                G.add_edge(block_id, neighbor_id)

    # Add POIs counts as node attributes
    poi_counts = blocks.set_index("block_id")["poi_count"].fillna(0).to_dict()
    nx.set_node_attributes(G, poi_counts, "poi_count")


    print(f"Graph constructed: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G