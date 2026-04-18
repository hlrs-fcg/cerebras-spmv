#!/usr/bin/env python3
"""
Execute notebooks and export all figures to images.
Usage:
    python run_and_export.py                    # Run all notebooks
    python run_and_export.py --recent-only      # Run only notebooks modified today
"""

import argparse
import subprocess
import json
from pathlib import Path
from datetime import datetime, timedelta
import base64

def get_notebooks(recent_only=False):
    """Get list of notebooks to process."""
    notebooks_dir = Path(__file__).parent
    notebooks = list(notebooks_dir.glob("*.ipynb"))

    # Filter out checkpoints
    notebooks = [nb for nb in notebooks if ".ipynb_checkpoints" not in str(nb)]

    if recent_only:
        # Only notebooks modified in last 24 hours
        cutoff = datetime.now() - timedelta(days=1)
        notebooks = [nb for nb in notebooks if datetime.fromtimestamp(nb.stat().st_mtime) > cutoff]

    return sorted(notebooks)

def execute_notebook(notebook_path):
    """Execute a notebook using jupyter nbconvert."""
    print(f"\n{'='*70}")
    print(f"Executing: {notebook_path.name}")
    print('='*70)

    try:
        result = subprocess.run(
            ['jupyter', 'nbconvert', '--to', 'notebook', '--execute',
             '--inplace', '--ExecutePreprocessor.timeout=600', str(notebook_path)],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Successfully executed {notebook_path.name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to execute {notebook_path.name}")
        print(f"Error: {e.stderr}")
        return False

def extract_images(notebook_path, output_dir):
    """Extract all image outputs from executed notebook."""
    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    notebook_name = notebook_path.stem
    img_count = 0

    for cell_idx, cell in enumerate(nb.get('cells', [])):
        if cell.get('cell_type') != 'code':
            continue

        outputs = cell.get('outputs', [])
        for output_idx, output in enumerate(outputs):
            # Check for image data in output
            data = output.get('data', {})

            if 'image/png' in data:
                img_data = data['image/png']
                img_bytes = base64.b64decode(img_data)

                img_filename = f"{notebook_name}_cell{cell_idx}_fig{output_idx}.png"
                img_path = output_dir / img_filename

                with open(img_path, 'wb') as img_file:
                    img_file.write(img_bytes)

                img_count += 1
                print(f"  → Saved {img_filename}")

    return img_count

def main():
    parser = argparse.ArgumentParser(description='Execute notebooks and export figures')
    parser.add_argument('--recent-only', action='store_true',
                       help='Only process notebooks modified in last 24 hours')
    args = parser.parse_args()

    # Setup
    notebooks_dir = Path(__file__).parent
    output_dir = notebooks_dir / 'figures'
    output_dir.mkdir(exist_ok=True)

    # Get notebooks to process
    notebooks = get_notebooks(args.recent_only)

    if not notebooks:
        print("No notebooks found to process")
        return

    print(f"Found {len(notebooks)} notebook(s) to process")
    print(f"Output directory: {output_dir}")

    # Execute and extract
    total_images = 0
    successful = 0

    for notebook in notebooks:
        if execute_notebook(notebook):
            successful += 1
            img_count = extract_images(notebook, output_dir)
            total_images += img_count
            print(f"  Extracted {img_count} image(s)")

    # Summary
    print(f"\n{'='*70}")
    print(f"Summary:")
    print(f"  Notebooks executed: {successful}/{len(notebooks)}")
    print(f"  Total images extracted: {total_images}")
    print(f"  Images saved to: {output_dir}")
    print('='*70)

if __name__ == '__main__':
    main()
