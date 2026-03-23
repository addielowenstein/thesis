#!/usr/bin/env bash
# Unzip PPLSDAILY zips and organize contents by date (from first 时间：YYYY-MM-DD in index.html)
set -e
SRC="/Users/addielowenstein/Downloads/PPLSDAILY"
ROOT="/Users/addielowenstein/thesis"
ARTICLES="$ROOT/articles"
TMP="$ROOT/_tmp_unzip"
mkdir -p "$ARTICLES"
mkdir -p "$TMP"
cd "$ROOT"

for z in "$SRC"/*.zip; do
  [ -f "$z" ] || continue
  base=$(basename "$z" .zip)
  echo "Processing $base ..."
  rm -rf "$TMP/$base"
  unzip -o -q "$z" -d "$TMP/$base"
  html="$TMP/$base/index.html"
  if [ ! -f "$html" ]; then
    echo "  No index.html, skipping"
    continue
  fi
  # First occurrence of 时间：YYYY-MM-DD
  date_line=$(grep -oE '时间：[0-9]{4}-[0-9]{2}-[0-9]{2}' "$html" | head -1)
  if [ -z "$date_line" ]; then
    echo "  No date found, using 0000/00/00"
    date_line="时间：0000-00-00"
  fi
  # 时间：2015-10-26 -> 2015 10 26
  ymd="${date_line#时间：}"
  y="${ymd%%-*}"      # 2015 (longest suffix match)
  rest="${ymd#*-}"    # 10-26
  m="${rest%-*}"      # 10
  d="${rest#*-}"      # 26
  outdir="$ARTICLES/$y/$m/$d/$base"
  mkdir -p "$outdir"
  cp -R "$TMP/$base"/* "$outdir/"
  echo "  -> $outdir"
done

echo "Cleaning up temp ..."
rm -rf "$TMP"
echo "Done. Articles are in $ARTICLES"
