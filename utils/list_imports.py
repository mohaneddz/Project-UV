import os
import ast
from pathlib import Path

def extract_imports(file_path):
    """Extract import statements from a Python file."""
    imports = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tree = ast.parse(f.read(), filename=file_path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"from {module} import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
    return imports

def main():
    # Get the directory of this script
    script_dir = Path(__file__).parent
    all_imports = []
    # Walk through all Python files
    for root, dirs, files in os.walk(script_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = Path(root) / file
                imports = extract_imports(file_path)
                if imports:
                    print(f"\n--- Imports in {file_path.relative_to(script_dir)} ---")
                    for imp in imports:
                        print(imp)
                        all_imports.append(imp)

    print(f"\n--- All unique imports ({len(set(all_imports))} total) ---")
    for imp in sorted(set(all_imports)):
        print(imp)

if __name__ == "__main__":
    main()