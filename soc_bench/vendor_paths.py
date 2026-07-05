"""Where pinned upstream method repos live once fetched via
`scripts/fetch_method.sh`, plus the exact commit each adapter is pinned to.

Pinning matters: a method's reported numbers should never silently change
because the upstream repo changed. Bump the commit here deliberately.
"""
from pathlib import Path

EXTERNAL = Path(__file__).resolve().parents[1] / "external"

PINNED = {
    "trust_region_soc": {
        "dirname": "TrustRegionSOC",
        "url": "https://github.com/DenisBless/TrustRegionSOC",
        "commit": "a93471f8698ab4754edc810184390769828ad4fa",
    },
}


def vendored_path(method: str) -> Path:
    info = PINNED[method]
    path = EXTERNAL / info["dirname"]
    if not path.exists():
        raise FileNotFoundError(
            f"'{method}' isn't vendored yet. Run:\n"
            f"  scripts/fetch_method.sh {method}\n"
            f"(clones {info['url']} @ {info['commit']})"
        )
    return path
