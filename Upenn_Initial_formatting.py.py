import os

def generate_directory_markdown(path, file):
    """
    Generate markdown format of the directory structure.

    Args:
    path (str): The directory path
    file (file): The file object to write to
    """
    for root, dirs, files in os.walk(path):
        level = root.replace(path, '').count(os.sep)
        indent = ' ' * 4 * level
        dir_name = os.path.basename(root) + '/'
        print(f"Directory: {dir_name}")  # Debugging print
        file.write(f"{indent}- {dir_name}\n")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"File: {f}")  # Debugging print
            file.write(f"{subindent}- {f}\n")

def main():
    path = input("Enter the path of the folder: ").rstrip(os.sep)
    output_file = "directory_structure.md"

    if os.path.isdir(path):
        with open(output_file, "w") as file:
            generate_directory_markdown(path, file)
        print(f"Markdown file '{output_file}' has been created.")
    else:
        print("Invalid directory path. Please check the path and try again.")

if __name__ == "__main__":
    main()
