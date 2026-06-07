#!/bin/bash
# Build all dashboard caches from the clean dataset
# Copy-paste vào terminal để chạy

cd "$(dirname "$0")" || exit 1

echo "=== Building all dashboard caches from clean dataset ==="
echo ""

CLEAN_CSV="Dataset/clean/ai_works_clean.csv"
CACHE_DIR="Dataset/dashboard_cache"

if [ ! -f "$CLEAN_CSV" ]; then
    echo "❌ Error: $CLEAN_CSV not found!"
    echo "Please run: python src/preprocess/merge_raw_dataset.py ... && python src/preprocess/build_clean_dataset.py ..."
    exit 1
fi

echo "📊 Step 1: Build core dashboard cache..."
python src/preprocess/build_core_dashboard_cache.py \
    --input "$CLEAN_CSV" \
    --output-dir "$CACHE_DIR"

echo ""
echo "📈 Step 2: Build impact ML cache (this takes ~5-10 minutes)..."
python src/preprocess/build_impact_ml_cache.py \
    --input "$CLEAN_CSV" \
    --output-dir "$CACHE_DIR"

echo ""
echo "🏢 Step 3: Build institution type view..."
python src/preprocess/build_institution_type_view.py \
    --inputs "$CLEAN_CSV" \
    --outdir "$CACHE_DIR"

echo ""
echo "📈 Step 4: Build rising/fading terms (this takes ~2-5 minutes)..."
python src/preprocess/build_rising_fading_terms_family.py \
    --input "$CLEAN_CSV" \
    --output-dir "$CACHE_DIR"

echo ""
echo "✅ All caches built successfully!"
echo "Cache directory: $CACHE_DIR"
ls -lh "$CACHE_DIR" | tail -20
