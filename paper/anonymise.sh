#!/usr/bin/env bash
# Produce an anonymised copy of the repo for double-blind submission.
# ICBINB-BIO anonymity extends to LINKED MATERIAL, so no public repo, org,
# institution or personal address may survive anywhere in the tree.
# Usage:  bash paper/anonymise.sh  ->  ../esp-lab-anon/
set -euo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)"; DST="${SRC}-anon"
rm -rf "$DST"; mkdir -p "$DST"

# copy tracked files only, minus the private and identifying ones.
#   paper/anonymise.sh  - contains the very strings it scrubs
#   outreach/           - drafts addressed to a named lab
#   polaris/            - cluster runbook; names the institution in every file
cd "$SRC"
git ls-files | grep -vE '^(paper/anonymise\.sh|outreach/|polaris/)' | while read -r f; do
  mkdir -p "$DST/$(dirname "$f")"; cp "$f" "$DST/$f"
done
cd "$DST"

# scrub identifying strings from every text file we ship. The extension list
# must cover EVERY tracked text type: a packaging file that lists the author's
# name and email is exactly as identifying as the paper's title page, and
# .toml was missing from this list until 18 Aug.
find . -type f \( -name '*.md' -o -name '*.py' -o -name '*.tex' -o -name '*.txt' \
     -o -name '*.json' -o -name '*.toml' -o -name '*.cfg' -o -name '*.ini' \
     -o -name '*.sh' -o -name '*.yml' -o -name '*.yaml' -o -name '*.sty' \
     -o -name '*.bib' -o -name '*.csv' -o -name '*.html' -o -name 'LICENSE' \
     -o -name '.gitignore' \) -print0 |
  xargs -0 sed -i '' \
    -e 's|github\.com/slug-labz/Audio-Jungle-Book-Project|ANONYMISED-REPO|g' \
    -e 's|github\.com/agentwolf27/esp-lab|ANONYMISED-REPO|g' \
    -e 's|Audio-Jungle-Book-Project|anon-repo|g' \
    -e 's|slug-labz|anon-org|g' \
    -e 's|agentwolf27|anon|g' \
    -e 's|Vishrut Malhotra|Anonymous Author|g' \
    -e 's|vishrutmalhotra[0-9]*@gmail\.com|anon@example.com|g' \
    -e 's|polaris\.sfsu\.edu|cluster.example.edu|g' \
    -e 's|SF State|the institution|g' \
    -e 's|SFSU|the institution|g' -e 's|sfsu|anon-inst|g' \
    -e 's|/Users/vish|/home/anon|g'

# no git history (it carries the author name and email on every commit)
rm -rf .git

echo "=== residual identifying strings (should be empty) ==="
# Scanned in python because the artifacts embed base64 images: a megabyte-long
# data: URI both drowns the output and throws false positives when a random
# base64 run happens to spell one of these. Strip the payloads, then scan.
python3 - <<'EOF' || { echo "!!! ANONYMISATION FAILED -- do not upload this tree" >&2; exit 1; }
import pathlib, re, sys
PAT = re.compile(r'agentwolf|vishrut|malhotra|slug-?labz|jungle-book|sfsu|sf state|polaris|/Users/vish',
                 re.I)
DATA = re.compile(r'data:[a-z/+.\-]*;base64,[A-Za-z0-9+/=]+', re.I)
bad = 0
for f in sorted(pathlib.Path('.').rglob('*')):
    if not f.is_file():
        continue
    try:
        text = f.read_text(errors='ignore')
    except OSError:
        continue
    for i, line in enumerate(DATA.sub('DATA-URI', text).splitlines(), 1):
        m = PAT.search(line)
        if m:
            print(f"  {f}:{i}: ...{line[max(0, m.start() - 40):m.end() + 40].strip()}...")
            bad += 1
sys.exit(1 if bad else 0)
EOF
echo "  clean"
echo "=== wrote $DST ($(find . -type f | wc -l | tr -d ' ') files) ==="
echo "Next: upload to anonymous.4open.science (or a fresh anonymous GitHub account),"
echo "then replace [ANONYMISED MIRROR URL] in paper_a.tex."
