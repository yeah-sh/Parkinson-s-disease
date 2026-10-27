"""Design tokens and the stylesheet of NeuroLab.

The colours match app/.streamlit/config.toml. Chart colours were checked with the data-viz
palette rules on the card surface (dark band, colour-blind separation, contrast): ``SERIES``
as a two-slot categorical pair, ``Q_RAMP`` as an ordinal ramp.
"""

from __future__ import annotations

BG = "#0B1020"
SURFACE = "#111A2D"
SURFACE_2 = "#151F33"
SURFACE_3 = "#1B2740"
BORDER = "#1F2B42"
BORDER_STRONG = "#2A3954"
TEXT = "#E6EDF7"
TEXT_2 = "#A3B1C6"
TEXT_3 = "#7C8BA3"
CYAN = "#22D3EE"
TEAL = "#5EEAD4"
AMBER = "#F5B544"
RED = "#F26D6D"

# Chart chrome.
GRID = "#1C2840"
AXIS = "#2A3954"
# Marks that give context (the recording before cleaning, a greyed-out model).
CONTEXT = "#5B6A85"
# Two series that are told apart: the positive class (or "towards it") and the negative one.
SERIES = ("#1a9fc0", "#d65f7e")
# Moment orders q, most negative to most positive: one hue, dim to bright.
Q_RAMP = ("#1a5566", "#17809b", "#22aacb", "#5fd0e8", "#b0eef9")
# Maps: magnitude dim to bright; polarity rose below zero, cyan above, slate at zero.
SEQUENTIAL = ("#12283a", "#1a5566", "#17809b", "#22aacb", "#5fd0e8", "#b0eef9")
DIVERGING = ("#f4a6b8", "#d65f7e", "#7d3550", "#2c3547", "#135a70", "#1a9fc0", "#9fe6f6")

FONT = '"Source Sans 3", "Source Sans Pro", "Segoe UI", system-ui, sans-serif'
MONO = '"Source Code Pro", ui-monospace, Consolas, monospace'

CSS = f"""
<style>
:root {{
  --nl-bg: {BG}; --nl-s1: {SURFACE}; --nl-s2: {SURFACE_2}; --nl-s3: {SURFACE_3};
  --nl-border: {BORDER}; --nl-border-2: {BORDER_STRONG};
  --nl-text: {TEXT}; --nl-text-2: {TEXT_2}; --nl-text-3: {TEXT_3};
  --nl-cyan: {CYAN}; --nl-teal: {TEAL}; --nl-amber: {AMBER}; --nl-red: {RED};
  --nl-radius: 10px; --nl-radius-sm: 6px;
  --nl-mono: {MONO};
}}

/* ---- shell ------------------------------------------------------------------------- */
[data-testid="stMainBlockContainer"] {{
  padding-top: 1.6rem; padding-bottom: 5rem; max-width: 1360px;
}}
header[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stSidebar"] {{ border-right: 1px solid var(--nl-border); }}
[data-testid="stSidebarHeader"] {{ padding-bottom: 0.25rem; }}
[data-testid="stSidebarNavSeparator"] {{ display: none; }}
[data-testid="stNavSectionHeader"] {{
  font-size: 0.66rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--nl-text-3); font-weight: 600; margin-top: 0.6rem;
}}
[data-testid="stSidebarNavLink"] {{ border-radius: var(--nl-radius-sm); }}
[data-testid="stSidebarNavLink"][aria-current="page"] {{
  background: rgba(34, 211, 238, 0.08);
  box-shadow: inset 2px 0 0 var(--nl-cyan);
}}
[data-testid="stSidebarNavLink"][aria-current="page"] span {{ color: var(--nl-text); }}

h1, h2, h3, h4 {{ letter-spacing: -0.012em; }}
h1 {{ font-size: 1.7rem !important; }}
h2 {{ font-size: 1.25rem !important; }}
h3 {{ font-size: 1.05rem !important; }}

/* ---- cards: keyed containers ------------------------------------------------------- */
[class*="st-key-nl-card"] {{
  background: var(--nl-s1); border: 1px solid var(--nl-border);
  border-radius: var(--nl-radius); padding: 1.05rem 1.2rem 1.15rem 1.2rem;
}}
[class*="st-key-nl-well"] {{
  background: var(--nl-s2); border: 1px solid var(--nl-border);
  border-radius: var(--nl-radius); padding: 0.85rem 1rem;
}}
[class*="st-key-nl-greyed"] {{ opacity: 0.5; filter: grayscale(1); }}
[class*="st-key-nl-header"] {{
  border-bottom: 1px solid var(--nl-border); padding-bottom: 0.9rem; margin-bottom: 0.4rem;
}}
[class*="st-key-nl-disclaimer"] {{
  border: 1px solid rgba(245, 181, 68, 0.28); border-left: 3px solid var(--nl-amber);
  background: rgba(245, 181, 68, 0.05); border-radius: var(--nl-radius-sm);
  padding: 0.6rem 0.75rem;
}}
[class*="st-key-nl-disclaimer"] p {{ font-size: 0.78rem; color: var(--nl-text-2);
  line-height: 1.45; }}

/* ---- native elements ------------------------------------------------------------- */
[data-testid="stMetric"] {{
  background: var(--nl-s1); border: 1px solid var(--nl-border);
  border-radius: var(--nl-radius); padding: 0.7rem 0.9rem;
}}
[data-testid="stMetricLabel"] p {{
  font-size: 0.7rem; letter-spacing: 0.09em; text-transform: uppercase;
  color: var(--nl-text-3); font-weight: 600;
}}
[data-testid="stMetricValue"] {{ font-family: var(--nl-mono); }}
[data-testid="stExpander"] details {{ border-color: var(--nl-border); background: var(--nl-s1); }}
[data-testid="stExpander"] summary p {{ font-weight: 600; }}
[data-testid="stCaptionContainer"] {{ color: var(--nl-text-3); }}
[data-testid="stFileUploaderDropzone"] {{
  background: var(--nl-s2); border: 1px dashed var(--nl-border-2); border-radius: var(--nl-radius);
  padding: 1.6rem 1.2rem;
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: var(--nl-cyan); }}
.stTabs [data-baseweb="tab-list"] {{ gap: 1.4rem; border-bottom: 1px solid var(--nl-border); }}
.stTabs [data-baseweb="tab"] {{ padding-left: 0; padding-right: 0; }}
.stTabs [data-baseweb="tab"] p {{ font-weight: 600; font-size: 0.9rem; }}
[data-testid="stAlertContainer"] {{ border-radius: var(--nl-radius-sm); }}
[data-testid="stVegaLiteChart"], [data-testid="stArrowVegaLiteChart"] {{ background: transparent; }}

/* ---- NeuroLab components (st.html) ----------------------------------------------- */
.nl-eyebrow {{
  font-size: 0.68rem; letter-spacing: 0.16em; text-transform: uppercase;
  color: var(--nl-cyan); font-weight: 650; margin: 0 0 0.2rem 0;
}}
.nl-title {{ font-size: 1.6rem; font-weight: 650; letter-spacing: -0.015em; color: var(--nl-text);
  margin: 0; line-height: 1.2; }}
.nl-lede {{ color: var(--nl-text-2); font-size: 0.95rem; margin: 0.3rem 0 0 0; max-width: 62rem;
  line-height: 1.5; }}
.nl-muted {{ color: var(--nl-text-3); }}
.nl-num {{ font-family: var(--nl-mono); font-variant-numeric: tabular-nums; }}
.nl-section {{ font-size: 0.7rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: var(--nl-text-3); font-weight: 650; margin: 0.2rem 0 0.55rem 0; }}

.nl-chips {{ display: flex; flex-wrap: wrap; gap: 6px; justify-content: flex-end; }}
.nl-chip {{
  display: inline-flex; align-items: center; gap: 6px; height: 24px; padding: 0 10px;
  border-radius: 999px; border: 1px solid var(--nl-border-2); background: var(--nl-s1);
  font-size: 0.74rem; color: var(--nl-text-2); white-space: nowrap; max-width: 100%;
  overflow: hidden; text-overflow: ellipsis;
}}
.nl-chip .nl-dot {{ width: 6px; height: 6px; border-radius: 50%; background: currentColor;
  flex: none; }}
.nl-chip.ok {{ color: var(--nl-teal); border-color: rgba(94, 234, 212, 0.32); }}
.nl-chip.warn {{ color: var(--nl-amber); border-color: rgba(245, 181, 68, 0.38); }}
.nl-chip.err {{ color: var(--nl-red); border-color: rgba(242, 109, 109, 0.4); }}
.nl-chip.run {{ color: var(--nl-cyan); border-color: rgba(34, 211, 238, 0.35); }}
.nl-chip.mute {{ color: var(--nl-text-3); }}
.nl-chip.research {{ color: var(--nl-amber); background: rgba(245, 181, 68, 0.06);
  border-color: rgba(245, 181, 68, 0.3); letter-spacing: 0.08em; text-transform: uppercase;
  font-size: 0.66rem; font-weight: 650; }}

.nl-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 10px; }}
.nl-tile {{ background: var(--nl-s1); border: 1px solid var(--nl-border);
  border-radius: var(--nl-radius); padding: 0.75rem 0.9rem; min-height: 86px; }}
.nl-tile .k {{ font-size: 0.68rem; letter-spacing: 0.1em; text-transform: uppercase;
  color: var(--nl-text-3); font-weight: 650; }}
.nl-tile .v {{ font-size: 1.35rem; font-weight: 600; color: var(--nl-text); margin-top: 0.25rem;
  font-family: var(--nl-mono); letter-spacing: -0.01em; }}
.nl-tile .v.empty {{ color: var(--nl-text-3); }}
.nl-tile .s {{ font-size: 0.78rem; color: var(--nl-text-3); margin-top: 0.15rem;
  line-height: 1.35; }}
.nl-tile.warn {{ border-color: rgba(245, 181, 68, 0.4); }}
.nl-tile.warn .v {{ color: var(--nl-amber); }}

.nl-steps {{ list-style: none; margin: 0; padding: 0; }}
.nl-step {{ display: grid; grid-template-columns: 26px 1fr auto; gap: 10px; align-items: start;
  position: relative; padding-bottom: 14px; }}
.nl-step:not(:last-child)::before {{ content: ""; position: absolute; left: 12px; top: 24px;
  bottom: 2px; width: 1px; background: var(--nl-border-2); }}
.nl-step .ic {{ width: 24px; height: 24px; border-radius: 50%; display: grid; place-items: center;
  border: 1px solid var(--nl-border-2); background: var(--nl-s2); color: var(--nl-text-3);
  font-size: 0.7rem; font-family: var(--nl-mono); }}
.nl-glyph {{ font-weight: 700; line-height: 1; font-family: var(--nl-mono); }}
.nl-step .t {{ font-weight: 600; color: var(--nl-text); font-size: 0.9rem; line-height: 24px; }}
.nl-step .d {{ color: var(--nl-text-3); font-size: 0.8rem; line-height: 1.4; margin-top: 1px; }}
.nl-step .m {{ font-size: 0.72rem; color: var(--nl-text-3); line-height: 24px;
  white-space: nowrap; }}
.nl-step.done .ic {{ color: var(--nl-teal); border-color: rgba(94, 234, 212, 0.45);
  background: rgba(94, 234, 212, 0.08); }}
.nl-step.done .m {{ color: var(--nl-teal); }}
.nl-step.failed .ic {{ color: var(--nl-red); border-color: rgba(242, 109, 109, 0.5);
  background: rgba(242, 109, 109, 0.08); }}
.nl-step.failed .m {{ color: var(--nl-red); }}
.nl-step.attention .ic {{ color: var(--nl-amber); border-color: rgba(245, 181, 68, 0.5);
  background: rgba(245, 181, 68, 0.08); }}
.nl-step.attention .m {{ color: var(--nl-amber); }}
.nl-step.ready .ic {{ color: var(--nl-cyan); border-color: rgba(34, 211, 238, 0.5); }}
.nl-step.ready .m {{ color: var(--nl-cyan); }}
.nl-step.running .ic {{ border-color: rgba(34, 211, 238, 0.25); border-top-color: var(--nl-cyan);
  animation: nl-spin 0.9s linear infinite; }}
.nl-step.running .m, .nl-step.running .d {{ color: var(--nl-cyan); }}
@keyframes nl-spin {{ to {{ transform: rotate(360deg); }} }}
@media (prefers-reduced-motion: reduce) {{ .nl-step.running .ic {{ animation: none; }} }}
.nl-steps.row {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; }}
.nl-steps.row .nl-step {{ grid-template-columns: 26px 1fr; padding-bottom: 0; }}
.nl-steps.row .nl-step::before {{ display: none; }}
.nl-steps.row .nl-step .m {{ grid-column: 2; line-height: 1.3; white-space: normal; }}

.nl-checks {{ display: flex; flex-direction: column; gap: 6px; }}
.nl-check {{ display: grid; grid-template-columns: 22px 130px 1fr; gap: 10px; align-items: start;
  padding: 0.6rem 0.75rem; border-radius: var(--nl-radius-sm); background: var(--nl-s2);
  border: 1px solid var(--nl-border); }}
.nl-check .ic {{ width: 20px; height: 20px; display: grid; place-items: center; }}
.nl-check .ic {{ font-size: 0.85rem; }}
.nl-check .n {{ font-weight: 600; color: var(--nl-text); font-size: 0.86rem;
  text-transform: capitalize; }}
.nl-check .r {{ font-size: 0.74rem; color: var(--nl-text-3); margin-top: 2px; line-height: 1.35; }}
.nl-check .x {{ font-size: 0.84rem; color: var(--nl-text-2); line-height: 1.45;
  overflow-wrap: anywhere; }}
.nl-check .fix {{ margin-top: 0.35rem; font-size: 0.8rem; color: var(--nl-text); }}
.nl-check.passed .ic {{ color: var(--nl-teal); }}
.nl-check.note .ic {{ color: var(--nl-amber); }}
.nl-check.failed {{ border-color: rgba(242, 109, 109, 0.45);
  background: rgba(242, 109, 109, 0.05); }}
.nl-check.failed .ic {{ color: var(--nl-red); }}
.nl-check.pending .ic {{ color: var(--nl-text-3); }}
.nl-check.pending .n {{ color: var(--nl-text-3); }}

.nl-kv {{ padding: 0; display: grid; grid-template-columns: minmax(120px, max-content) 1fr;
  column-gap: 18px;
  row-gap: 6px; font-size: 0.86rem; margin: 0; }}
.nl-kv dt {{ color: var(--nl-text-3); }}
.nl-kv dd {{ margin: 0; color: var(--nl-text); overflow-wrap: anywhere; }}

.nl-file {{ display: grid; grid-template-columns: 40px 1fr auto; gap: 12px; align-items: center; }}
.nl-file .badge {{ width: 40px; height: 40px; border-radius: 8px; display: grid;
  place-items: center;
  background: var(--nl-s3); border: 1px solid var(--nl-border-2); font-family: var(--nl-mono);
  font-size: 0.68rem; color: var(--nl-cyan); font-weight: 600; }}
.nl-file .name {{ font-weight: 600; color: var(--nl-text); overflow-wrap: anywhere;
  font-size: 0.92rem; }}
.nl-file .meta {{ font-size: 0.78rem; color: var(--nl-text-3); margin-top: 1px; }}

.nl-modules {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 10px; }}
.nl-module {{ background: var(--nl-s1); border: 1px solid var(--nl-border);
  border-radius: var(--nl-radius);
  padding: 0.85rem 0.95rem; }}
.nl-module .h {{ display: flex; justify-content: space-between; align-items: center; gap: 8px; }}
.nl-module .t {{ font-weight: 600; color: var(--nl-text); font-size: 0.92rem; }}
.nl-module .d {{ font-size: 0.8rem; color: var(--nl-text-3); margin-top: 0.3rem;
  line-height: 1.4; }}

.nl-provenance {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 8px; }}
.nl-prov {{ border: 1px solid var(--nl-border); border-radius: var(--nl-radius-sm);
  padding: 0.6rem 0.75rem;
  background: var(--nl-s2); opacity: 0.55; }}
.nl-prov.active {{ opacity: 1; border-color: rgba(34, 211, 238, 0.55);
  background: rgba(34, 211, 238, 0.06); }}
.nl-prov .t {{ font-weight: 600; font-size: 0.84rem; color: var(--nl-text); display: flex; gap: 6px;
  align-items: center; }}
.nl-prov .d {{ font-size: 0.76rem; color: var(--nl-text-3); margin-top: 0.2rem; line-height: 1.4; }}

.nl-legend {{ display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.78rem;
  color: var(--nl-text-2); }}
.nl-legend span {{ display: inline-flex; align-items: center; gap: 6px; }}
.nl-legend i {{ display: inline-block; width: 12px; height: 8px; border-radius: 2px; }}
.nl-legend i.line {{ height: 2px; border-radius: 1px; }}

.nl-empty {{ border: 1px dashed var(--nl-border-2); border-radius: var(--nl-radius);
  padding: 2rem 1.4rem;
  text-align: center; background: var(--nl-s1); }}
.nl-empty .t {{ font-weight: 600; color: var(--nl-text); font-size: 1rem; }}
.nl-empty .d {{ color: var(--nl-text-3); font-size: 0.86rem; margin: 0.35rem auto 0 auto;
  max-width: 34rem;
  line-height: 1.5; }}

abbr.nl-term {{ text-decoration: underline dotted var(--nl-text-3); text-underline-offset: 3px;
  cursor: help; }}

@media (max-width: 900px) {{
  .nl-steps.row {{ grid-template-columns: 1fr; }}
  .nl-check {{ grid-template-columns: 22px 1fr; }}
  .nl-check .x {{ grid-column: 2; }}
  .nl-chips {{ justify-content: flex-start; }}
}}
</style>
"""
