import os
"""
Util script to convert folder with documents of various formats into text files
"""

def convert_to_txt_and_delete_old(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Construct the full file path
            file_path = os.path.join(root, file)
            # Split the file name and its extension
            file_base = os.path.splitext(file_path)[0]
            # New file path with .txt extension
            new_file_path = file_base + '.txt'
            # Rename the file (this deletes the old file)
            os.rename(file_path, new_file_path)
            print(f"Converted {file_path} to {new_file_path}")


convert_to_txt_and_delete_old("lang_chain_docs_2024")
