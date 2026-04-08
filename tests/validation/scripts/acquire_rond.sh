#!/usr/bin/env bash
# Acquire the Mayo Clinic Radiation Oncology NLP Database (ROND).
#
# License: Apache 2.0 (redistributable)
# Source: https://github.com/Mayo-Clinic-RadOnc-Foundation-Models/Radiation-Oncology-NLP-Database
#
# Run from repo root:
#   bash tests/validation/scripts/acquire_rond.sh

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"

# Source local environment if present (copied from env.example.sh).
# All shell scripts in this repository follow this convention so that
# machine-specific paths (PIPER_VOICES_ROOT, PIPER_BIN, etc.) are
# picked up from a single source of truth. See README.md §
# "Quick setup: env.example.sh" for the copy-and-source workflow.
if [ -f "${REPO_ROOT}/env.sh" ]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/env.sh"
fi

DEST="${REPO_ROOT}/tests/validation/corpora/redistributable/rond/raw"

if [ -d "${DEST}/.git" ]; then
    echo "ROND already cloned at ${DEST}"
    echo "Updating..."
    git -C "${DEST}" pull --ff-only
else
    echo "Cloning ROND to ${DEST}..."
    mkdir -p "$(dirname "${DEST}")"
    git clone --depth=1 \
        https://github.com/Mayo-Clinic-RadOnc-Foundation-Models/Radiation-Oncology-NLP-Database.git \
        "${DEST}"
fi

echo ""
echo "ROND acquired successfully."
echo "  Location: ${DEST}"
echo "  Next step: python tests/validation/scripts/extract_rond_text.py"
