import os

def generate_tree(dir_path, prefix=""):
    items = sorted(os.listdir(dir_path))
    # Exclude some directories like .git, .trae, __pycache__, etc.
    items = [i for i in items if i not in ('.git', '.trae', '__pycache__')]
    
    tree_str = ""
    for i, item in enumerate(items):
        path = os.path.join(dir_path, item)
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        tree_str += f"{prefix}{connector}{item}\n"
        
        if os.path.isdir(path):
            extension = "    " if is_last else "│   "
            # Limit depth or specific folders if too large
            if item in ('ultralytics', 'demo', 'datasets', 'build'):
                tree_str += f"{prefix}{extension}...\n"
            else:
                tree_str += generate_tree(path, prefix + extension)
    return tree_str

with open("tree.txt", "w", encoding="utf-8") as f:
    f.write(generate_tree("."))
