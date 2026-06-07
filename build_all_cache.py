#!/usr/bin/env python
"""
Build all dashboard caches from the clean dataset in one go.

Usage:
    python build_all_cache.py
    
Or with custom paths:
    python build_all_cache.py --clean-csv Dataset/clean/ai_works_clean.csv --cache-dir Dataset/dashboard_cache
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, name):
    """Run a command and report status."""
    print(f"\n{'='*60}")
    print(f"🚀 {name}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"❌ {name} failed with return code {result.returncode}")
        return False
    print(f"✅ {name} completed successfully")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-csv",
        default="Dataset/clean/ai_works_clean.csv",
        help="Path to clean CSV file"
    )
    parser.add_argument(
        "--cache-dir",
        default="Dataset/dashboard_cache",
        help="Output directory for cache files"
    )
    args = parser.parse_args()
    
    clean_csv = Path(args.clean_csv)
    cache_dir = Path(args.cache_dir)
    
    # Verify clean CSV exists
    if not clean_csv.exists():
        print(f"❌ Error: {clean_csv} not found!")
        print("\nPlease run first:")
        print("  python src/preprocess/merge_raw_dataset.py ...")
        print("  python src/preprocess/build_clean_dataset.py ...")
        sys.exit(1)
    
    print(f"\n📂 Using clean CSV: {clean_csv}")
    print(f"📂 Cache output: {cache_dir}\n")
    
    builders = [
        (
            ["python", "src/preprocess/build_core_dashboard_cache.py",
             "--input", str(clean_csv),
             "--output-dir", str(cache_dir)],
            "Step 1: Core dashboard cache"
        ),
        (
            ["python", "src/preprocess/build_impact_ml_cache.py",
             "--input", str(clean_csv),
             "--output-dir", str(cache_dir)],
            "Step 2: Impact ML cache for Tab 3 (⏱️  ~5-10 mins)"
        ),
        (
            ["python", "src/preprocess/build_institution_type_view.py",
             "--inputs", str(clean_csv),
             "--outdir", str(cache_dir)],
            "Step 3: Institution type view"
        ),
        (
            ["python", "src/preprocess/build_rising_fading_terms_family.py",
             "--input", str(clean_csv),
             "--output-dir", str(cache_dir)],
            "Step 4: Rising/fading terms (⏱️  ~2-5 mins)"
        ),
    ]
    
    all_passed = True
    for cmd, name in builders:
        if not run_command(cmd, name):
            all_passed = False
            break
    
    print(f"\n{'='*60}")
    if all_passed:
        print("✅ All caches built successfully!")
        print(f"{'='*60}\n")
        cache_files = sorted(cache_dir.glob("*.csv"))
        print(f"📊 Generated {len(cache_files)} cache files:\n")
        for f in cache_files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"  • {f.name:<45} ({size_mb:>8.1f} MB)")
    else:
        print("❌ Cache build failed - check output above")
        print(f"{'='*60}")
        sys.exit(1)


if __name__ == "__main__":
    main()
