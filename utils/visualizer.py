"""
Matplotlib chart helpers for the Streamlit UI.
All figures are rendered to an in-memory BytesIO PNG buffer
so they can be passed directly to st.image().
"""

import io

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')   # headless backend — no display required

# Brand palette (matches .streamlit/config.toml)
_BG_DARK    = '#0E1117'
_PANEL_DARK = '#1A1C24'
_ACCENT     = '#6C63FF'
_TEXT       = '#FAFAFA'


def plot_keyword_chart(keywords: list[tuple[str, float]]) -> io.BytesIO:
    """
    Horizontal bar chart of the top-10 keywords by TF-IDF score.

    Parameters
    ----------
    keywords : list of (term, score) tuples, sorted by score descending.

    Returns
    -------
    BytesIO PNG buffer ready for st.image().
    """
    top = keywords[:10]
    words  = [k for k, _ in top]
    scores = [s for _, s in top]

    fig, ax = plt.subplots(figsize=(8, 4))

    # Dark theme
    fig.patch.set_facecolor(_BG_DARK)
    ax.set_facecolor(_PANEL_DARK)

    # Bars — reversed so highest score appears at top
    bars = ax.barh(words[::-1], scores[::-1], color=_ACCENT, height=0.6)

    # Value labels on bars
    for bar, score in zip(bars, scores[::-1]):
        ax.text(
            bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
            f"{score:.4f}",
            va='center', ha='left', fontsize=8, color=_TEXT
        )

    ax.set_xlabel('TF-IDF Score', color=_TEXT, fontsize=9)
    ax.set_title('Top Keywords by Importance', color=_TEXT, fontsize=11, pad=10)
    ax.tick_params(colors=_TEXT, labelsize=9)
    ax.xaxis.label.set_color(_TEXT)

    for spine in ax.spines.values():
        spine.set_edgecolor('#333344')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def plot_summary_stats(
    total_sentences: int,
    selected_sentences: int,
    word_count: int,
) -> io.BytesIO:
    """
    Small donut chart showing compression ratio alongside text stats.

    Returns
    -------
    BytesIO PNG buffer.
    """
    kept     = selected_sentences
    discarded = max(total_sentences - selected_sentences, 0)

    fig, (ax_donut, ax_text) = plt.subplots(
        1, 2, figsize=(7, 3.5),
        gridspec_kw={'width_ratios': [1, 1]}
    )
    fig.patch.set_facecolor(_BG_DARK)

    # ── Donut ────────────────────────────────────────────────────────────
    ax_donut.set_facecolor(_BG_DARK)
    sizes  = [kept, discarded] if discarded > 0 else [1, 0]
    colors = [_ACCENT, '#2E2E42']
    wedges, _ = ax_donut.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops=dict(width=0.45, edgecolor=_BG_DARK)
    )
    pct = round(kept / max(total_sentences, 1) * 100)
    ax_donut.text(
        0, 0, f"{pct}%", ha='center', va='center',
        fontsize=16, fontweight='bold', color=_TEXT
    )
    ax_donut.set_title("Sentences kept", color=_TEXT, fontsize=9, pad=6)

    # ── Text stats ───────────────────────────────────────────────────────
    ax_text.set_facecolor(_BG_DARK)
    ax_text.axis('off')
    stats = [
        ("Total sentences",    str(total_sentences)),
        ("Summary sentences",  str(selected_sentences)),
        ("Word count",         str(word_count)),
        ("Compression",        f"{100 - pct}%"),
    ]
    for row, (label, value) in enumerate(stats):
        ax_text.text(0.05, 0.85 - row * 0.22, label,
                     color='#AAAACC', fontsize=9, transform=ax_text.transAxes)
        ax_text.text(0.65, 0.85 - row * 0.22, value,
                     color=_TEXT, fontsize=10, fontweight='bold',
                     transform=ax_text.transAxes)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf
