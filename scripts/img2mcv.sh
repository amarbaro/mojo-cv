#!/usr/bin/env bash
# img2mcv.sh IN OUT.mcv [gray|bgr]: decode any image ImageMagick reads (JPEG, PNG, ...) into MCV1 uint8,
# EXIF orientation applied; an embedded non-sRGB profile (Display P3) is converted to sRGB. Header: "MCV1", u32 rows, u32 cols, u16 ch, u16 dtype 0 (uint8), little-endian.
set -euo pipefail
[ $# -ge 2 ] || { echo "FAIL usage: img2mcv.sh IN OUT.mcv [gray|bgr]" >&2; exit 2; }
in=$1 out=$2 mode=${3:-gray}
case $mode in gray) ch=1 fmt=gray cs=(-colorspace Gray) ;; bgr) ch=3 fmt=bgr cs=() ;; *) echo "FAIL mode: $mode" >&2; exit 2 ;; esac
# ImageMagick exits non-zero on harmless XMP warnings (iPhone files), so judge by the output, not the status
wh=$(magick "$in" -auto-orient -format '%w %h' info: 2>/dev/null || true)
read -r w h <<< "$wh"
# embedded non-sRGB profile (iPhone Display P3): convert to sRGB so values mean the same across cameras
SRGB=${SRGB_ICC:-/usr/share/ghostscript/iccprofiles/srgb.icc}; icc=()
desc=$(magick "$in" -format '%[icc:description]' info: 2>/dev/null || true)
if [ -n "$desc" ] && ! printf '%s' "$desc" | grep -qi 'srgb'; then [ -f "$SRGB" ] || { echo "FAIL icc: $SRGB missing for $desc" >&2; exit 1; }; icc=(-profile "$SRGB"); fi
 [ -n "${h:-}" ] || { echo "FAIL decode: $in" >&2; exit 1; }
le32() { printf "\\x$(printf %02x $(($1 & 255)))\\x$(printf %02x $(($1 >> 8 & 255)))\\x$(printf %02x $(($1 >> 16 & 255)))\\x$(printf %02x $(($1 >> 24 & 255)))"; }
{ printf 'MCV1'; le32 "$h"; le32 "$w"; printf "\\x$(printf %02x $ch)\\x00\\x00\\x00"
  magick "$in" -auto-orient "${icc[@]}" "${cs[@]}" -depth 8 "$fmt:-" 2>/dev/null || true; } > "$out"
[ "$(stat -c %s "$out")" -eq $((16 + w * h * ch)) ] || { echo "FAIL size: $out" >&2; exit 1; }
