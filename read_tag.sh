#!/usr/bin/env bash
# Read a merchandise tag photo with the vision model.
#
#     ./read_tag.sh ~/Downloads/IMG_4013.HEIC
#
# Converts the HEIC to JPEG, shrinks it to 2016px on the long side, writes the
# request body to a file (the image is too big for a command line), and posts it.
# The proxy caches identical requests and answers a repeat in a millisecond, so
# the body asks for no cache - otherwise the timer measures the cache, not the
# model. Remove "cache":{"no-cache":true} to get the cached answer back.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${1:?usage: ./read_tag.sh photo.HEIC}"

"$HERE/.venv/bin/python" "$HERE/scripts/heic_to_jpg.py" "$IMG" /tmp/tag.jpg -q 92
sips -Z 2016 /tmp/tag.jpg >/dev/null

PROMPT='This is a photo of a Gap garment merchandise tag. Read it and return ONE markdown table with two columns, Field and Value, one row per field in this order: File name; Style number (the 6-digit number); Style-colour number (the 9-digit number); Colour code (3 digits); Colour name; Biz Unit; ISD (only what is printed after ISD:, or - if nothing); Code on the line below ISD; INDC date; Product name; Fit note; Status selected (which of Shoot, CC, Lev has its circle filled); Status date; Division; Department; Class; Shot codes (comma-separated); Other colours (comma-separated, add TICKED after any with a ticked box); Barcode number; Stamps (red ink text); Stickers; Unreadable (anything you could not read). Write every value exactly as printed. Use - for a blank. Do not guess. Return the table only, no other text.'

{
  printf '%s' '{"model":"vision-qwen","temperature":0,"max_tokens":1500,"cache":{"no-cache":true},"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":[{"type":"text","text":"'"$PROMPT"'"},{"type":"image_url","image_url":{"url":"data:image/jpeg;base64,'
  base64 < /tmp/tag.jpg | tr -d '\n'
  printf '%s' '"}}]}]}'
} > /tmp/body.json

T0=$(python3 -c 'import time; print(time.time())')
curl -s http://10.11.245.145:4000/v1/chat/completions \
  -H "Authorization: Bearer $(sed -n 's/^QWEN_API_KEY *= *//p' "$HERE/.env")" \
  -H "Content-Type: application/json" \
  --data-binary @/tmp/body.json \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])"
T1=$(python3 -c 'import time; print(time.time())')
python3 -c "print(f'\n[model call {$T1 - $T0:.1f}s]')"
