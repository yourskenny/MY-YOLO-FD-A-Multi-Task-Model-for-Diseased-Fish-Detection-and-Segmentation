import os

def generate_tree(dir_path, prefix="", depth=0, max_depth=2):
    items = sorted(os.listdir(dir_path))
    items = [i for i in items if i not in ('.git', '.trae', '__pycache__')]
    
    tree_str = ""
    for i, item in enumerate(items):
        path = os.path.join(dir_path, item)
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        tree_str += f"{prefix}{connector}{item}\n"
        
        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            if depth < max_depth:
                tree_str += generate_tree(path, prefix + extension, depth + 1, max_depth)
            else:
                tree_str += f"{prefix}{extension}...\n"
    return tree_str

with open("tree_ultralytics.txt", "w", encoding="utf-8") as f:
    f.write(generate_tree("ultralytics"))
