import os
import urllib.request
import zipfile
import sys

def download_file(url, target_path):
    print(f"Downloading from {url} to {target_path}...")
    def report(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        if count % 1000 == 0:
            print(f"  {percent}% ({count * block_size / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", flush=True)
    urllib.request.urlretrieve(url, target_path, reporthook=report)
    print(f"Downloaded {target_path} successfully.")

def unzip_file(zip_path, extract_dir):
    print(f"Extracting {zip_path} to {extract_dir}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Extracted successfully to {extract_dir}.")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "scierc_models")
    os.makedirs(model_dir, exist_ok=True)

    models = [
        # Single-sentence models (W=0)
        ("ent-scib-ctx0", "https://nlp.cs.princeton.edu/projects/pure/scierc_models/ent-scib-ctx0.zip"),
        ("rel-scib-ctx0", "https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel-scib-ctx0.zip"),
        ("rel_approx-scib-ctx0", "https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel_approx-scib-ctx0.zip"),
        # Cross-sentence models (W=300 for entity, W=100 for relation)
        ("ent-scib-ctx300", "https://nlp.cs.princeton.edu/projects/pure/scierc_models/ent-scib-ctx300.zip"),
        ("rel-scib-ctx100", "https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel-scib-ctx100.zip"),
        ("rel_approx-scib-ctx100", "https://nlp.cs.princeton.edu/projects/pure/scierc_models/rel_approx-scib-ctx100.zip")
    ]

    for name, url in models:
        target_dir = os.path.join(model_dir, name)
        if os.path.exists(target_dir):
            print(f"Model directory {target_dir} already exists, skipping.")
            continue
        zip_file = os.path.join(model_dir, f"{name}.zip")
        download_file(url, zip_file)
        unzip_file(zip_file, model_dir)
        if os.path.exists(zip_file):
            os.remove(zip_file)

if __name__ == "__main__":
    main()
