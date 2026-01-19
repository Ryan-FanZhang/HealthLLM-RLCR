#!/bin/bash
# =============================================================================
# Healthcare Data Processing Script
# =============================================================================
# This script processes all healthcare datasets and optionally uploads to HuggingFace Hub
#
# Usage:
#   ./run_all_data_processing.sh                    # Only save locally
#   ./run_all_data_processing.sh --push             # Save locally + upload to HF Hub
#   ./run_all_data_processing.sh --push --user xxx  # Specify HF username
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Configuration
# =============================================================================

# HuggingFace username (change this to your username)
HF_USER="RyanFan"

# Whether to push to HuggingFace Hub (default: false)
PUSH_TO_HUB=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH_TO_HUB=true
            shift
            ;;
        --user)
            HF_USER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--push] [--user HF_USERNAME]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Print Configuration
# =============================================================================

echo "=============================================================="
echo "Healthcare Data Processing"
echo "=============================================================="
echo "HuggingFace User: ${HF_USER}"
echo "Push to Hub: ${PUSH_TO_HUB}"
echo "=============================================================="
echo ""

# =============================================================================
# HuggingFace Auth Check (only when --push)
# =============================================================================

if [ "$PUSH_TO_HUB" = true ]; then
    echo "Note: Data will be saved locally AND uploaded to HuggingFace Hub"
    echo ""
    
    echo "Checking HuggingFace login status..."

    # 优先使用新 CLI: hf auth whoami
    if command -v hf &> /dev/null; then
        if ! hf auth whoami > /dev/null 2>&1; then
            echo "Error: Not logged in to HuggingFace."
            echo "Please run: hf auth login"
            exit 1
        fi
    # 兼容旧版本: huggingface-cli whoami
    elif command -v huggingface-cli &> /dev/null; then
        if ! huggingface-cli whoami > /dev/null 2>&1; then
            echo "Error: Not logged in to HuggingFace."
            echo "Please run: huggingface-cli login"
            exit 1
        fi
    else
        echo "Error: HuggingFace CLI not found."
        echo "Install it via: pip install huggingface_hub"
        echo "Then login with: hf auth login"
        exit 1
    fi

    echo "HuggingFace login OK!"
    echo ""
fi

# =============================================================================
# Process PMData
# =============================================================================

echo "=============================================================="
echo "[1/4] Processing PMData..."
echo "=============================================================="

if [ "$PUSH_TO_HUB" = true ]; then
    python data/creation_scripts/pmdata_healthcare.py \
        --push_to_hub \
        --hub_name "${HF_USER}/pmdata"
else
    python data/creation_scripts/pmdata_healthcare.py
fi

echo "PMData processing complete!"
echo ""

# =============================================================================
# Process LifeSnaps
# =============================================================================

echo "=============================================================="
echo "[2/4] Processing LifeSnaps..."
echo "=============================================================="

if [ "$PUSH_TO_HUB" = true ]; then
    python data/creation_scripts/lifesnaps_healthcare.py \
        --push_to_hub \
        --hub_name "${HF_USER}/lifesnaps"
else
    python data/creation_scripts/lifesnaps_healthcare.py
fi

echo "LifeSnaps processing complete!"
echo ""

# =============================================================================
# Process GLOBEM
# =============================================================================

echo "=============================================================="
echo "[3/4] Processing GLOBEM..."
echo "=============================================================="

if [ "$PUSH_TO_HUB" = true ]; then
    python data/creation_scripts/globem_healthcare.py \
        --push_to_hub \
        --hub_name "${HF_USER}/globem"
else
    python data/creation_scripts/globem_healthcare.py
fi

echo "GLOBEM processing complete!"
echo ""

# =============================================================================
# Process AW_FB
# =============================================================================

echo "=============================================================="
echo "[4/4] Processing AW_FB..."
echo "=============================================================="

if [ "$PUSH_TO_HUB" = true ]; then
    python data/creation_scripts/awfb_healthcare.py \
        --push_to_hub \
        --hub_name "${HF_USER}/awfb"
else
    python data/creation_scripts/awfb_healthcare.py
fi

echo "AW_FB processing complete!"
echo ""

# =============================================================================
# Summary
# =============================================================================

echo "=============================================================="
echo "All datasets processed successfully!"
echo "=============================================================="
echo ""
echo "Local data saved to:"
echo "  - ./data/pmdata_processed/"
echo "  - ./data/lifesnaps_processed/"
echo "  - ./data/globem_processed/"
echo "  - ./data/awfb_processed/"
echo ""

if [ "$PUSH_TO_HUB" = true ]; then
    echo "HuggingFace Hub datasets (expected):"
    echo "  - ${HF_USER}/pmdata_gen, ${HF_USER}/pmdata_tac, ${HF_USER}/pmdata_tabc, ${HF_USER}/pmdata_tabc_long"
    echo "  - ${HF_USER}/lifesnaps_gen, ${HF_USER}/lifesnaps_tac, ${HF_USER}/lifesnaps_tabc, ${HF_USER}/lifesnaps_tabc_long"
    echo "  - ${HF_USER}/globem_gen, ${HF_USER}/globem_tac, ${HF_USER}/globem_tabc, ${HF_USER}/globem_tabc_long"
    echo "  - ${HF_USER}/awfb_gen, ${HF_USER}/awfb_tac, ${HF_USER}/awfb_tabc, ${HF_USER}/awfb_tabc_long"
    echo ""
fi

echo "To load a dataset:"
echo "  from datasets import load_from_disk"
echo "  dataset = load_from_disk('./data/pmdata_processed/pmdata_tabc')"
echo ""
echo "=============================================================="
