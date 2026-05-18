def choose_tree_template(version):
    tree_template = None
    if version == "general":
        tree_template = [[0], [1], [2], [3], [4], [5], [6], [7],
                        [0,0], [0,1], [0,2], [0,3], [0,4], [0,5], [0,6], [0,7], [1,0], [1,1], [1,2], [1,3], [2,0], [2,1], [2,2], [3,0], [3,1], [4,0], [5,0], [6,0], [7,0],
                        [0,0,0], [0,0,1], [0,0,2], [0,0,3], [0,0,4], [0,0,5], [0,0,6], [0,0,7], [0,1,0], [0,1,1], [0,1,2], [0,2,0], [0,2,1], [0,3,0], [0,4,0], [0,5,0], [0,6,0], [0,7,0], [1,0,0], [1,0,1], [1,1,0], [2,0,0], [3,0,0], [4,0,0], [5,0,0],
                        [0,0,0,0], [0,0,0,1], [0,0,0,2], [0,0,0,3], [0,0,0,4], [0,0,1,0], [0,0,1,1], [0,0,2,0], [0,0,3,0], [0,0,4,0], [0,1,0,0], [0,2,0,0], [1,0,0,0], [2,0,0,0], [3,0,0,0],
                        [0,0,0,0,0], [0,0,0,0,1], [0,0,0,0,2], [0,0,0,1,0], [0,0,0,2,0], [0,0,1,0,0], [0,1,0,0,0], [1,0,0,0,0],
                        [0,0,0,0,0,0], [0,0,0,0,0,1], [0,0,0,1,0,0],
                        ]
    if tree_template is None:
        raise ValueError("Invalid version")
    return tree_template


if __name__ == "__main__":
    tree_template = choose_tree_template("general")
    
    # Analyze paths: find the 6 root paths and their depths
    root_paths_info = {}
    for path in tree_template:
        if path and len(path) > 0:
            root = path[0]
            depth = len(path)
            if root not in root_paths_info:
                root_paths_info[root] = []
            root_paths_info[root].append(depth)
    
    # Print number of paths and depth of each path
    num_paths = len(root_paths_info)
    print(f"Number of paths: {num_paths}")
    print("Depth of each path:")
    for root in sorted(root_paths_info.keys()):
        depths = root_paths_info[root]
        max_depth = max(depths)
        min_depth = min(depths)
        num_subpaths = len(depths)
        print(f"  Path {root}: depth range [{min_depth}, {max_depth}], {num_subpaths} sub-paths")
    
    print(f"\nFull tree template ({len(tree_template)} nodes):") # (80 nodes)
    print(tree_template)