#!/usr/bin/env bash
# sync_public.sh — Push changes from private repo to public showcase repo.
# Usage: bash scripts/sync_public.sh ["optional commit message"]

set -euo pipefail

PRIVATE="/Users/shivam/Desktop/Programming/Quant"
PUBLIC="/Users/shivam/Desktop/Programming/QuantPublic"
MSG="${1:-Sync from private repo $(date +%Y-%m-%d)}"

echo "==> Syncing files to $PUBLIC ..."

rsync -av --delete \
  --exclude='.git/' \
  --exclude='.claude/' \
  --exclude='CLAUDE.md' \
  --exclude='BACKLOG.md' \
  --exclude='README.md' \
  --exclude='data/' \
  --exclude='memory/' \
  --exclude='explore_*.py' \
  --exclude='degiro_orders.py' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.db' \
  --exclude='*.log' \
  --exclude='*.lic' \
  "$PRIVATE/" "$PUBLIC/"

echo ""
echo "==> Applying substitutions ..."

# Email → env var placeholder
find "$PUBLIC" -name "*.py" -exec sed -i '' \
  's|"personal-research shivam3125@gmail.com"|os.getenv("EDGAR_IDENTITY", "your-name your@email.com")|g' {} +

find "$PUBLIC" -name "*.py" -exec sed -i '' \
  's|"universe-builder shivam3125@gmail.com"|os.getenv("EDGAR_IDENTITY", "your-name your@email.com")|g' {} +

# Personal Python path → sys.executable
find "$PUBLIC" -name "*.py" -exec sed -i '' \
  's|"/Users/shivam/opt/anaconda3/envs/quant/bin/python3.13"|sys.executable|g' {} +

echo ""
echo "==> Committing and pushing ..."
cd "$PUBLIC"
git add .

if git diff --cached --quiet; then
  echo "Nothing changed — public repo already up to date."
else
  git diff --cached --stat
  git commit -m "$MSG"
  git push
  echo ""
  echo "Done. Public repo updated."
fi
