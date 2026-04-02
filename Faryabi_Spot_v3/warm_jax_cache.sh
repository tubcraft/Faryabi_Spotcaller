#!/usr/bin/env bash
# warm_jax_cache.sh — Run once to compile Piscis for your image shape.
# After this, all future runs skip compilation and are fast.
#
# Usage:
#   bash warm_jax_cache.sh /path/to/one_image.tif

set -e

IMAGE="${1:-}"

# ── 1. Set JAX cache directory (no admin needed — user home) ──────────────
export JAX_COMPILATION_CACHE_DIR="${HOME}/.jax_cache"
export JAX_CACHE_DIR="${HOME}/.jax_cache"
mkdir -p "${HOME}/.jax_cache"
echo "JAX cache: ${HOME}/.jax_cache"

# ── 2. Add to ~/.bashrc so it persists permanently ────────────────────────
BASHRC="${HOME}/.bashrc"
if ! grep -q "JAX_COMPILATION_CACHE_DIR" "${BASHRC}" 2>/dev/null; then
    echo "" >> "${BASHRC}"
    echo "# JAX compilation cache — speeds up Piscis after first run" >> "${BASHRC}"
    echo "export JAX_COMPILATION_CACHE_DIR=${HOME}/.jax_cache" >> "${BASHRC}"
    echo "export JAX_CACHE_DIR=${HOME}/.jax_cache" >> "${BASHRC}"
    echo "Added JAX cache vars to ~/.bashrc"
else
    echo "~/.bashrc already has JAX cache vars"
fi

# ── 3. Check if cache already has a compiled kernel ───────────────────────
CACHED=$(find "${HOME}/.jax_cache" -type f 2>/dev/null | wc -l)
echo "Cache files: ${CACHED}"
if [ "${CACHED}" -gt 10 ]; then
    echo "Cache already warm — you are good to run with multiple workers."
    exit 0
fi

# ── 4. Run a single warm-up inference to populate the cache ───────────────
echo ""
echo "Running warm-up (1 worker, 1 image)..."
echo "This will take ~20 min the first time (JAX compiles for your image shape)."
echo "All future runs will be fast."
echo ""

if [ -z "${IMAGE}" ]; then
    echo "No image path provided. Set JAX_COMPILATION_CACHE_DIR and run manually:"
    echo "  export JAX_COMPILATION_CACHE_DIR=${HOME}/.jax_cache"
    echo "  python run_piscis.py --input_dir /your/data --output_dir /tmp/warmup \"
    echo "      --detector piscis --model 20230905 --threshold 0.5 \"
    echo "      --workers 1 --stack --run_only piscis"
    echo ""
    echo "After that run finishes, use --workers 2 or more for real runs."
else
    # Create a temp dir for warmup output
    TMPOUT=$(mktemp -d)
    trap "rm -rf ${TMPOUT}" EXIT

    # Get the input dir from the image path
    INDIR=$(dirname "${IMAGE}")

    JAX_COMPILATION_CACHE_DIR="${HOME}/.jax_cache"     JAX_CACHE_DIR="${HOME}/.jax_cache"     python run_piscis.py         --input_dir  "${INDIR}"         --output_dir "${TMPOUT}"         --detector   piscis         --model      20230905         --threshold  0.5         --workers    1         --stack         --run_only   piscis

    echo ""
    CACHED=$(find "${HOME}/.jax_cache" -type f 2>/dev/null | wc -l)
    echo "Cache now has ${CACHED} files."
    echo ""
    echo "Warm-up done! You can now run with --workers 2 or more."
fi
