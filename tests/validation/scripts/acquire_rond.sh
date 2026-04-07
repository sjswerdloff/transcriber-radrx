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
