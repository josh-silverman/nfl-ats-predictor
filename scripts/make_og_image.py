"""Draw the link-preview image (site/assets/og.png, 1200x630). Run once; rerun only if the branding changes.

    conda run -n nfl-ats python scripts/make_og_image.py

Deliberately generic (no record or numbers), so it never goes stale.
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from nfl_ats.data import PROJECT_ROOT  # noqa: E402

OUT = PROJECT_ROOT / "site" / "assets" / "og.png"
INK, INK2, MUTED, ACCENT, PAGE = "#111418", "#4a5361", "#7a8492", "#2a78d6", "#f4f6f8"


def main() -> None:
    fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor=PAGE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 630)
    ax.axis("off")

    ax.add_patch(FancyBboxPatch((60, 60), 1080, 510, boxstyle="round,pad=0,rounding_size=28", fc="white", ec="#e3e7ec", lw=2))
    ax.add_patch(FancyBboxPatch((110, 455), 54, 54, boxstyle="round,pad=0,rounding_size=12", fc=ACCENT, ec="none"))
    ax.add_patch(FancyBboxPatch((128, 473), 18, 18, boxstyle="round,pad=0,rounding_size=4", fc="white", ec="none"))
    ax.text(185, 482, "Josh's NFL Elo", fontsize=40, fontweight="bold", color=INK, va="center")
    ax.text(110, 395, "Who's likely to win each NFL game,", fontsize=27, color=INK2, va="center")
    ax.text(110, 352, "locked before kickoff and compared with Vegas.", fontsize=27, color=INK2, va="center")

    # A win-chance bar like the site's game cards: favorite solid, underdog faded, a dark tick for Vegas.
    left, width, y, h, p = 110, 980, 215, 34, 0.64
    ax.add_patch(FancyBboxPatch((left, y), width * (1 - p) - 3, h, boxstyle="round,pad=0,rounding_size=17",
                                fc="#eb6834", ec="none", alpha=0.35))
    ax.add_patch(FancyBboxPatch((left + width * (1 - p) + 3, y), width * p - 3, h, boxstyle="round,pad=0,rounding_size=17",
                                fc=ACCENT, ec="none"))
    vegas_x = left + width * (1 - 0.69)
    ax.add_patch(FancyBboxPatch((vegas_x - 4, y - 12), 8, h + 24, boxstyle="round,pad=0,rounding_size=4", fc=INK, ec="white", lw=3))
    ax.text(left, 165, "Away 36%", fontsize=20, color=MUTED, va="center")
    ax.text(left + width, 165, "Home 64%", fontsize=20, color=INK, va="center", ha="right", fontweight="bold")
    ax.text(vegas_x, 125, "▎Vegas", fontsize=16, color=MUTED, va="center", ha="center")

    fig.savefig(OUT, dpi=100, facecolor=PAGE)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
