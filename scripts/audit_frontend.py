"""Keep the frontend audit active while a specific upstream fix is unavailable."""
import json
import subprocess
import sys

result = subprocess.run(
    ["npm", "audit", "--json", "--audit-level=low"],
    capture_output=True,
    text=True,
    check=False,
)
if result.returncode not in (0, 1):
    print(result.stderr, file=sys.stderr)
    raise SystemExit("npm audit could not complete")

try:
    audit = json.loads(result.stdout)
    vulnerabilities = audit["vulnerabilities"]
except (ValueError, KeyError, TypeError):
    raise SystemExit("npm audit returned an invalid report")

allowed_url = "https://github.com/advisories/GHSA-fc5q-39xc-vjpf"
advisories = [
    item
    for vulnerability in vulnerabilities.values()
    for item in vulnerability["via"]
    if isinstance(item, dict)
]
if vulnerabilities and not advisories:
    raise SystemExit("npm audit returned vulnerabilities without advisory details")

unexpected = [item for item in advisories if item.get("url") != allowed_url]
if unexpected:
    for item in unexpected:
        print(f"Unexpected vulnerability: {item.get('name')} {item.get('url')}", file=sys.stderr)
    raise SystemExit(1)

if advisories:
    print(f"Known unpatched frontend development dependency advisory: {allowed_url}")
    print("Remove this exception when a patched picomatch version is available.")
else:
    print("No frontend vulnerabilities found. Remove the obsolete audit exception.")

