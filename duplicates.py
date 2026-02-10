import os
import hashlib
from PIL import Image
import imagehash
from collections import defaultdict

root_dir = ""
valid_extensions = ('.jpg', '.jpeg', '.png')

def get_hash(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def get_phash(path):
    img = Image.open(path).convert("RGB")
    return imagehash.phash(img)

def remove_duplicates(root_dir):
    hash_map = defaultdict(list)
    removed_files = []

    for folder, _, files in os.walk(root_dir):
        for file in files:
            if not file.lower().endswith(valid_extensions):
                continue
            path = os.path.join(folder, file)
            try:
                file_hash = get_hash(path)
                hash_map[file_hash].append(path)
            except Exception as e:
                print(f"Could not read {path}: {e}")

    for paths in hash_map.values():
        if len(paths) > 1:
            keep = paths[0]
            to_delete = paths[1:]
            for p in to_delete:
                try:
                    os.remove(p)
                    removed_files.append(p)
                except Exception as e:
                    print(f"Failed to delete {p}: {e}")
            print(f"\n Duplicates\n  Kept: {keep}\n  Deleted: {to_delete}")
    
    return removed_files

def remove_similar(root_dir, threshold=5):
    hash_map = {}
    removed = []

    for folder, _, files in os.walk(root_dir):
        for file in files:
            if not file.lower().endswith(valid_extensions):
                continue
            path = os.path.join(folder, file)
            try:
                phash = get_phash(path)

                similar = False
                for existing_path, existing_hash in hash_map.items():
                    if abs(phash - existing_hash) <= threshold:
                        os.remove(path)
                        removed.append(path)
                        print(f"Similar\n  Deleted: {path}\n  Kept: {existing_path}")
                        similar = True
                        break

                if not similar:
                    hash_map[path] = phash

            except Exception as e:
                print(f"Could not process {path}: {e}")
    return removed

duplicates = remove_duplicates(root_dir)
print(f"\n Removed {len(duplicates)}\n")

similar = remove_similar(root_dir, threshold=5)
print(f"\n Removed {len(similar)}\n")
