#!/bin/bash
# Convenience launcher: submit the full analysis pipeline with sensible
# dependency chaining so jobs only run once their inputs are ready.
#
# Usage:
#   bash submit_all.sh                   # submit everything
#   bash submit_all.sh embedding         # only embedding-based jobs (skip raw)
#   bash submit_all.sh raw               # only raw-pixel jobs (no embedding)
#   bash submit_all.sh rounds            # only per-round persistence
#
# You can also submit any single sbatch file by hand, e.g.
#   sbatch tsne.sbatch
#   sbatch persistence_rounds.sbatch
# These are completely independent of submit_all.sh.

set -euo pipefail

cd "$(dirname "$0")"

what="${1:-all}"
echo "submit_all.sh: target = $what"

submit() {
    local name="$1"; shift
    local jid
    jid=$(sbatch --parsable "$@" "$name")
    echo "  submitted $name -> $jid"
    echo "$jid"
}

case "$what" in
    all|embedding|labels)
        LABELS_JID=$(submit prepare_labels.sbatch)
        ;;
esac

case "$what" in
    all|embedding)
        # t-SNE / UMAP / persistence on SimCLR embeddings (3 backbones x 3 datasets each).
        submit tsne.sbatch       --dependency=afterok:${LABELS_JID:-}
        submit umap.sbatch       --dependency=afterok:${LABELS_JID:-}
        submit persistence.sbatch --dependency=afterok:${LABELS_JID:-}
        ;;
esac

case "$what" in
    all|raw)
        RAW_JID=$(submit extract_raw.sbatch)
        submit tsne_raw.sbatch --dependency=afterok:${RAW_JID}
        submit umap_raw.sbatch --dependency=afterok:${RAW_JID}
        ;;
esac

case "$what" in
    all|rounds)
        # Per-round persistence is independent of labels/raw.
        submit persistence_rounds.sbatch
        ;;
esac

echo "All requested jobs queued. Check with: squeue -u $USER"
