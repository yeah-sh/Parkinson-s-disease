"""Restwave's look: a calm night-time lab. Ink-navy page, off-white paper cards, electric teal.

Every colour is a token here and a CSS variable on the page. The chart colours were checked with
the data-viz palette rules on the card surface (lightness band, chroma floor, colour-blind
separation, contrast): the two groups are ``PD`` and ``HC``, adjacent ΔE 22.7 under protanopia
and deuteranopia. Text tokens meet WCAG AA on the surface they sit on. Verdict colours never
travel alone: every verdict has an icon and words.
"""

from __future__ import annotations

import altair as alt

# Page (ink) and its text.
INK = "#0B1322"
INK_2 = "#111C2E"
INK_3 = "#1C2A42"
ON_INK = "#E8EDF4"  # 15.8:1 on INK
ON_INK_2 = "#A9B6C8"  # 9.0:1
ON_INK_3 = "#8E9CB0"  # 6.7:1
TEAL = "#2EE6D0"  # the signature accent on ink, 11.8:1
RED_ON_INK = "#FF9A8F"

# Cards (paper) and their text.
PAPER = "#F6F3EC"
PAPER_2 = "#EDE8DD"
LINE = "#DDD6C8"
TEXT = "#17202C"  # 14.8:1 on PAPER
TEXT_2 = "#4A5566"  # 6.8:1
TEXT_3 = "#5B6573"  # 5.3:1
TEAL_DEEP = "#0B7C73"  # signals on paper, 4.6:1

# The two groups in every chart (marks only; labels stay in text colours).
PD = "#B86514"
HC = "#2B7BBE"
PD_WASH = "#F2DEC6"
HC_WASH = "#D6E4F1"

# Verdicts: edge, wash, text on the wash.
AMBER, AMBER_WASH, AMBER_TEXT = "#D98A1F", "#FBEBD0", "#7A420A"
SAGE, SAGE_WASH, SAGE_TEXT = "#6E9A73", "#E2ECDF", "#355E3F"
SLATE, SLATE_WASH, SLATE_TEXT = "#7A8698", "#E4E8EE", "#434E5E"
RED_TEXT = "#A3261B"

# Chart chrome on paper.
GRID = "#E6E0D4"
AXIS = "#CFC7B8"

DISPLAY = '"Fraunces", "Iowan Old Style", Georgia, serif'
BODY = '"IBM Plex Sans", "Segoe UI", system-ui, sans-serif'
MONO = '"IBM Plex Mono", ui-monospace, Consolas, monospace'
FONTS = (
    "https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,600"
    "&family=IBM+Plex+Mono:wght@500&family=IBM+Plex+Sans:wght@400;500;600&display=swap"
)

_TOKENS = {
    "ink": INK, "ink-2": INK_2, "ink-3": INK_3, "on-ink": ON_INK, "on-ink-2": ON_INK_2,
    "on-ink-3": ON_INK_3, "teal": TEAL, "red-on-ink": RED_ON_INK, "paper": PAPER,
    "paper-2": PAPER_2, "line": LINE, "text": TEXT, "text-2": TEXT_2, "text-3": TEXT_3,
    "teal-deep": TEAL_DEEP, "pd": PD, "hc": HC, "pd-wash": PD_WASH, "hc-wash": HC_WASH,
    "amber": AMBER, "amber-wash": AMBER_WASH, "amber-text": AMBER_TEXT, "sage": SAGE,
    "sage-wash": SAGE_WASH, "sage-text": SAGE_TEXT, "slate": SLATE, "slate-wash": SLATE_WASH,
    "slate-text": SLATE_TEXT, "red-text": RED_TEXT, "display": DISPLAY, "body": BODY,
    "mono": MONO,
}  # fmt: skip
_VARIABLES = " ".join(f"--rw-{name}: {value};" for name, value in _TOKENS.items())

CSS = f"""
<style>
@import url("{FONTS}");
:root {{ {_VARIABLES} --rw-radius: 18px; --rw-radius-sm: 10px; }}

/* ---- the page, without Streamlit's chrome --------------------------------------------- */
.stApp {{ background: var(--rw-ink); color: var(--rw-on-ink); font-family: var(--rw-body); }}
header[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stStatusWidget"], [data-testid="stMainMenu"], #MainMenu, footer,
[data-testid="stAppDeployButton"], [data-testid="stSidebar"],
[data-testid="stSidebarCollapsedControl"], [data-testid="stExpandSidebarButton"] {{
  display: none !important;
}}
[data-testid="stMainBlockContainer"] {{
  max-width: 1120px; padding: 2.2rem 1.5rem 4rem 1.5rem;
}}
.stApp p, .stApp label, .stApp button, .stApp input {{ font-family: var(--rw-body); }}
[data-testid="stVerticalBlock"] {{ gap: 1.1rem; }}
@media (max-width: 640px) {{
  [data-testid="stMainBlockContainer"] {{ padding: 1.2rem 1rem 3rem 1rem; }}
}}

/* ---- the hero ------------------------------------------------------------------------- */
.rw-brand {{ display: flex; align-items: center; justify-content: space-between; gap: 12px;
  flex-wrap: wrap; }}
.rw-logo {{ display: inline-flex; align-items: center; gap: 10px; font-family: var(--rw-display);
  font-weight: 600; font-size: 1.25rem; color: var(--rw-on-ink); letter-spacing: -0.01em; }}
.rw-logo img {{ flex: none; display: block; }}
.rw-tag {{ font-size: 0.78rem; color: var(--rw-on-ink-2); border: 1px solid var(--rw-ink-3);
  border-radius: 999px; padding: 4px 12px; }}
.rw-hero h1 {{ font-family: var(--rw-display); font-weight: 600; color: var(--rw-on-ink);
  font-size: clamp(2.1rem, 5vw, 3.4rem); line-height: 1.06; letter-spacing: -0.02em;
  margin: 2.4rem 0 0.9rem 0; max-width: 16ch; padding: 0; }}
.rw-hero .rw-lede {{ color: var(--rw-on-ink-2); font-size: 1.08rem; line-height: 1.6;
  max-width: 44rem; margin: 0; }}
.rw-wave {{ height: 56px; margin: 1.6rem 0 0.2rem 0; overflow: hidden; position: relative;
  -webkit-mask-image: linear-gradient(90deg, transparent, #000 12%, #000 88%, transparent);
  mask-image: linear-gradient(90deg, transparent, #000 12%, #000 88%, transparent); }}
.rw-wave img {{ width: 200%; height: 56px; display: block;
  animation: rw-drift 26s linear infinite; }}
@keyframes rw-drift {{ from {{ transform: translateX(0); }} to {{ transform: translateX(-50%); }} }}
@media (prefers-reduced-motion: reduce) {{ .rw-wave img {{ animation: none; }} }}

/* the upload zone: Streamlit's uploader, keyboard and screen-reader support kept */
.st-key-rw-drop [data-testid="stFileUploader"] {{ position: relative; }}
.st-key-rw-drop [data-testid="stWidgetLabel"] {{ position: absolute; top: 2.4rem; left: 1rem;
  right: 1rem; z-index: 1; pointer-events: none; display: flex; justify-content: center;
  margin: 0; }}
.st-key-rw-drop [data-testid="stWidgetLabel"] p {{ font-family: var(--rw-display);
  font-weight: 600; font-size: clamp(1.2rem, 2.6vw, 1.55rem); color: var(--rw-on-ink);
  text-align: center; margin: 0; }}
.st-key-rw-drop [data-testid="stFileUploader"] section,
.st-key-rw-drop [data-testid="stFileUploaderDropzone"] {{
  min-height: 210px; border: 1.5px dashed rgba(46, 230, 208, 0.55); border-radius: var(--rw-radius);
  background: radial-gradient(ellipse at 50% 0%, rgba(46, 230, 208, 0.10), transparent 70%),
    var(--rw-ink-2);
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  gap: 14px; padding: 5.6rem 1.25rem 2rem 1.25rem; text-align: center;
  transition: border-color 0.15s;
}}
.st-key-rw-drop [data-testid="stFileUploaderDropzone"]:hover,
.st-key-rw-drop [data-testid="stFileUploaderDropzone"]:focus-within {{
  border-color: var(--rw-teal); }}
.st-key-rw-drop [data-testid="stFileUploaderDropzoneInstructions"] {{
  display: flex; flex-direction: column; align-items: center; gap: 6px; margin: 0; }}
.st-key-rw-drop [data-testid="stFileUploaderDropzoneInstructions"] svg,
.st-key-rw-drop [data-testid="stFileUploaderDropzoneInstructions"] [data-testid="stIconMaterial"] {{
  color: var(--rw-teal); width: 40px; height: 40px; font-size: 40px; }}
.st-key-rw-drop [data-testid="stFileUploaderDropzoneInstructions"] span {{
  color: var(--rw-on-ink-3); font-size: 0.85rem; font-weight: 400; }}
.st-key-rw-drop [data-testid="stFileUploaderDropzoneInstructions"] small {{
  color: var(--rw-on-ink-3); font-size: 0.85rem; }}
.st-key-rw-drop button {{ background: var(--rw-teal); color: var(--rw-ink); border: 0;
  border-radius: 999px; font-weight: 600; padding: 0.5rem 1.3rem; }}
.st-key-rw-drop button:hover {{ background: #6FF0E0; color: var(--rw-ink); }}
.rw-drop-hint {{ color: var(--rw-on-ink-3); font-size: 0.88rem; margin: -0.3rem 0 0 0; }}

/* buttons on ink */
.st-key-rw-sample button, .st-key-rw-again button {{
  background: transparent; color: var(--rw-on-ink); border: 1px solid #33496B;
  border-radius: 999px; padding: 0.45rem 1.1rem; font-weight: 500; }}
.st-key-rw-sample button:hover, .st-key-rw-again button:hover {{
  border-color: var(--rw-teal); color: var(--rw-teal); }}
.st-key-rw-sample button:focus-visible, .st-key-rw-again button:focus-visible,
.st-key-rw-drop button:focus-visible {{ outline: 2px solid var(--rw-teal); outline-offset: 2px; }}
[data-testid="stCheckbox"] label p {{ color: var(--rw-on-ink); font-size: 0.95rem; }}

/* the strip that replaces the hero once a file is loaded */
.rw-strip {{ display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }}
.rw-files {{ display: flex; gap: 8px; flex-wrap: wrap; }}
.rw-file {{ font-family: var(--rw-mono); font-size: 0.82rem; color: var(--rw-on-ink-2);
  background: var(--rw-ink-2); border: 1px solid var(--rw-ink-3); border-radius: 8px;
  padding: 4px 10px; overflow-wrap: anywhere; }}
.rw-file b {{ color: var(--rw-teal); font-weight: 500; margin-right: 6px; }}

/* ---- the input check ------------------------------------------------------------------ */
.rw-chips {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
.rw-chips .rw-of {{ color: var(--rw-on-ink-3); font-size: 0.8rem; margin-right: 4px;
  font-family: var(--rw-mono); }}
.rw-chip {{ display: inline-flex; align-items: center; gap: 8px; padding: 6px 13px 6px 9px;
  border-radius: 999px; font-size: 0.9rem; color: var(--rw-on-ink);
  background: var(--rw-ink-2); border: 1px solid var(--rw-ink-3); }}
.rw-chip .k {{ color: var(--rw-on-ink-3); font-size: 0.78rem; }}
.rw-chip img {{ flex: none; display: block; }}
.rw-chip.passed svg {{ color: var(--rw-teal); }}
.rw-chip.failed {{ border-color: rgba(255, 154, 143, 0.55); color: var(--rw-red-on-ink); }}
.rw-chip.failed .k {{ color: var(--rw-red-on-ink); }}
.rw-chip.pending {{ color: var(--rw-on-ink-3); }}

/* ---- processing --------------------------------------------------------------------- */
.rw-steps {{ background: var(--rw-ink-2); border: 1px solid var(--rw-ink-3);
  border-radius: var(--rw-radius); padding: 1.1rem 1.3rem; }}
.rw-steps ol {{ list-style: none; margin: 0; padding: 0; display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }}
.rw-steps li {{ display: flex; align-items: center; gap: 10px; color: var(--rw-on-ink-3);
  font-size: 0.92rem; position: relative; }}
.rw-steps .dot {{ width: 26px; height: 26px; border-radius: 50%; flex: none; display: grid;
  place-items: center; border: 1.5px solid #33496B; }}
.rw-steps li.done {{ color: var(--rw-on-ink-2); }}
.rw-steps li.done .dot {{ background: var(--rw-teal); border-color: var(--rw-teal);
  color: var(--rw-ink); }}
.rw-steps li.active {{ color: var(--rw-on-ink); font-weight: 500; }}
.rw-steps li.active .dot {{ border-color: var(--rw-teal);
  box-shadow: 0 0 0 4px rgba(46, 230, 208, 0.16); }}
.rw-steps li.active .dot::after {{ content: ""; width: 10px; height: 10px; border-radius: 50%;
  background: var(--rw-teal); animation: rw-pulse 1.4s ease-in-out infinite; }}
@keyframes rw-pulse {{ 50% {{ opacity: 0.35; transform: scale(0.75); }} }}
@media (prefers-reduced-motion: reduce) {{ .rw-steps li.active .dot::after {{ animation: none; }} }}
.rw-steps .now {{ margin: 0.85rem 0 0 0; color: var(--rw-on-ink-2); font-size: 0.9rem; }}
@media (max-width: 760px) {{ .rw-steps ol {{ grid-template-columns: 1fr 1fr; }} }}

/* ---- paper cards ---------------------------------------------------------------------- */
.rw-card, [class*="st-key-rw-card"] {{
  background: var(--rw-paper); color: var(--rw-text); border: 1px solid var(--rw-line);
  border-radius: var(--rw-radius); padding: 1.5rem 1.6rem;
}}
[class*="st-key-rw-card"] {{ gap: 0.6rem; }}
.rw-card h2, .rw-card h3, [class*="st-key-rw-card"] h3 {{ color: var(--rw-text); padding: 0; }}
.rw-eyebrow {{ font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase;
  font-weight: 600; margin: 0 0 0.35rem 0; color: var(--rw-text-3); }}
.rw-title {{ font-family: var(--rw-display); font-weight: 600; font-size: 1.3rem;
  margin: 0; line-height: 1.25; color: var(--rw-text); letter-spacing: -0.01em; }}
.rw-caption {{ color: var(--rw-text-2); font-size: 0.95rem; line-height: 1.55;
  margin: 0.4rem 0 0 0; }}
.rw-caption b {{ color: var(--rw-text); font-weight: 600; }}
.rw-legend {{ display: flex; flex-wrap: wrap; gap: 6px 16px; font-size: 0.85rem;
  color: var(--rw-text-2); margin: 0.2rem 0 0 0; }}
.rw-legend span {{ display: inline-flex; align-items: center; gap: 7px; }}
.rw-legend i {{ display: inline-block; width: 16px; height: 8px; border-radius: 4px; }}
.rw-legend i.dot {{ width: 10px; height: 10px; border-radius: 50%;
  box-shadow: 0 0 0 2px var(--rw-paper); }}
.rw-legend i.diamond {{ width: 9px; height: 9px; border-radius: 1px; transform: rotate(45deg); }}
.rw-legend i.line {{ height: 3px; border-radius: 2px; }}
.rw-legend i.dash {{ height: 0; border-top: 3px dashed; border-radius: 0;
  background: none !important; }}
.rw-missing {{ color: var(--rw-text-3); font-size: 0.9rem; background: var(--rw-paper-2);
  border-radius: var(--rw-radius-sm); padding: 0.8rem 1rem; margin: 0.4rem 0 0 0; }}
[class*="st-key-rw-card"] [data-testid="stVegaLiteChart"] {{ background: transparent; }}
.rw-section {{ font-family: var(--rw-display); font-weight: 600; font-size: 1.55rem;
  color: var(--rw-on-ink); margin: 1.4rem 0 0 0; letter-spacing: -0.01em; }}
.rw-section-lede {{ color: var(--rw-on-ink-2); margin: 0.3rem 0 0 0; font-size: 0.98rem; }}

/* ---- the verdict -------------------------------------------------------------------- */
.rw-verdict {{ position: relative; overflow: hidden; padding: 1.8rem 2rem 1.6rem 2.3rem; }}
.rw-verdict::before {{ content: ""; position: absolute; inset: 0 auto 0 0; width: 8px;
  background: var(--edge); }}
.rw-verdict.detected {{ --edge: var(--rw-amber); --wash: var(--rw-amber-wash);
  --say: var(--rw-amber-text); }}
.rw-verdict.clear {{ --edge: var(--rw-sage); --wash: var(--rw-sage-wash);
  --say: var(--rw-sage-text); }}
.rw-verdict.inconclusive, .rw-verdict.none {{ --edge: var(--rw-slate); --wash: var(--rw-slate-wash);
  --say: var(--rw-slate-text); }}
.rw-verdict-grid {{ display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 1.4rem 2rem;
  align-items: center; }}
.rw-verdict-head {{ display: flex; gap: 1.1rem; align-items: flex-start; }}
.rw-icon {{ width: 64px; height: 64px; border-radius: 50%; flex: none; display: grid;
  place-items: center; background: var(--wash); color: var(--say); }}
.rw-verdict .rw-eyebrow {{ color: var(--say); }}
.rw-headline {{ font-family: var(--rw-display); font-weight: 600; color: var(--rw-text);
  font-size: clamp(1.9rem, 4.2vw, 2.9rem); line-height: 1.05; letter-spacing: -0.02em;
  margin: 0; padding: 0; }}
.rw-verdict.small .rw-headline {{ font-size: clamp(1.45rem, 3vw, 2rem); }}
.rw-verdict.small .rw-icon {{ width: 52px; height: 52px; }}
.rw-body {{ color: var(--rw-text-2); font-size: 1.02rem; line-height: 1.55; margin: 0.7rem 0 0 0;
  max-width: 40rem; }}
.rw-trust {{ margin: 1.3rem 0 0 0; padding: 0.85rem 1.05rem; border-radius: var(--rw-radius-sm);
  background: var(--rw-paper-2); color: var(--rw-text-2); font-size: 0.97rem; line-height: 1.55; }}
.rw-trust b {{ color: var(--rw-text); }}
.rw-trust.low b {{ color: var(--rw-amber-text); }}
.rw-note {{ margin: 0.9rem 0 0 0; color: var(--rw-text-2); font-size: 0.93rem; }}
.rw-gauge {{ margin: 0; text-align: center; }}
.rw-gauge img {{ width: 100%; max-width: 300px; height: auto; display: block; margin: 0 auto; }}
.rw-gauge .scale {{ display: grid; grid-template-columns: 1fr auto 1fr; align-items: start;
  max-width: 300px; margin: 0.25rem auto 0 auto; font-size: 0.85rem; color: var(--rw-text-2); }}
.rw-gauge .scale span:first-child {{ text-align: left; }}
.rw-gauge .scale span:last-child {{ text-align: right; }}
.rw-gauge .value {{ font-size: 2rem; font-weight: 600; color: var(--rw-text); line-height: 1;
  display: flex; flex-direction: column; align-items: center; gap: 4px; margin-top: -0.35rem; }}
.rw-gauge .value small {{ font-size: 0.78rem; font-weight: 400; color: var(--rw-text-3); }}
.rw-gauge figcaption {{ font-size: 0.82rem; color: var(--rw-text-3); margin-top: 0.3rem;
  line-height: 1.45; }}
.rw-pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.6rem; }}
.rw-pair > div {{ min-width: 0; }}
.rw-pair .rw-verdict-grid {{ grid-template-columns: 1fr; }}
.rw-which {{ font-family: var(--rw-mono); font-size: 0.8rem; color: var(--rw-text-3);
  margin: 0 0 0.5rem 0; overflow-wrap: anywhere; }}
@media (max-width: 860px) {{
  .rw-verdict-grid, .rw-pair {{ grid-template-columns: 1fr; }}
  .rw-verdict {{ padding: 1.4rem 1.2rem 1.3rem 1.5rem; }}
  .rw-icon {{ width: 48px; height: 48px; }}
}}

/* notes above the verdict, and the stop message */
.rw-banner {{ display: flex; gap: 12px; align-items: flex-start; background: var(--rw-ink-2);
  border: 1px solid var(--rw-ink-3); border-left: 3px solid var(--rw-teal);
  border-radius: var(--rw-radius-sm); padding: 0.8rem 1rem; color: var(--rw-on-ink-2);
  font-size: 0.93rem; line-height: 1.5; }}
.rw-banner img {{ flex: none; margin-top: 1px; }}
.rw-banner b {{ color: var(--rw-on-ink); font-weight: 600; }}
.rw-stop {{ border-left: 8px solid var(--rw-red-text); }}
.rw-stop .rw-headline {{ font-size: clamp(1.5rem, 3.4vw, 2.1rem); }}
.rw-stop .rw-eyebrow {{ color: var(--rw-red-text); }}
.rw-stop .fix {{ margin: 0.8rem 0 0 0; color: var(--rw-text); font-size: 1rem; line-height: 1.55; }}
.rw-stop .fix b {{ font-weight: 600; }}

/* ---- what drove the decision ------------------------------------------------------- */
.rw-drivers {{ list-style: none; margin: 0.6rem 0 0 0; padding: 0; display: grid; gap: 14px; }}
.rw-drivers li {{ display: grid; gap: 5px; }}
.rw-drivers .name {{ font-size: 0.95rem; color: var(--rw-text); line-height: 1.35; }}
.rw-drivers .row {{ display: grid; grid-template-columns: 1fr 1fr; align-items: center;
  height: 14px; position: relative; }}
.rw-drivers .row::after {{ content: ""; position: absolute; left: 50%; top: -3px; bottom: -3px;
  width: 1px; background: var(--rw-text-3); }}
.rw-drivers .bar {{ height: 12px; }}
.rw-drivers .left {{ display: flex; justify-content: flex-end; }}
.rw-drivers .left .bar {{ background: var(--rw-hc); border-radius: 4px 0 0 4px;
  margin-right: 1px; }}
.rw-drivers .right .bar {{ background: var(--rw-pd); border-radius: 0 4px 4px 0;
  margin-left: 1px; }}
.rw-drivers .dir {{ font-size: 0.8rem; color: var(--rw-text-3); }}
.rw-drivers .dir.towards-right {{ text-align: right; }}
.rw-axis {{ display: grid; grid-template-columns: 1fr 1fr; font-size: 0.8rem;
  color: var(--rw-text-3);
  margin: 0.4rem 0 0 0; }}
.rw-axis span:last-child {{ text-align: right; }}

/* ---- footer ------------------------------------------------------------------------- */
.rw-footer {{ border-top: 1px solid var(--rw-ink-3); padding-top: 1.2rem; margin-top: 0.6rem;
  color: var(--rw-on-ink-3); font-size: 0.9rem; line-height: 1.6; }}
.rw-footer a {{ color: var(--rw-teal); text-decoration: none; font-weight: 500; }}
.rw-footer a:hover {{ text-decoration: underline; }}
.rw-footer code {{ font-family: var(--rw-mono); font-size: 0.82rem; color: var(--rw-on-ink-2);
  background: var(--rw-ink-2); padding: 1px 6px; border-radius: 5px; }}
.st-key-rw-download button {{ background: var(--rw-paper); color: var(--rw-text); border: 0;
  border-radius: 999px; font-weight: 600; padding: 0.55rem 1.3rem; }}
.st-key-rw-download button:hover {{ background: #FFFFFF; color: var(--rw-text); }}
.st-key-rw-download button:focus-visible {{ outline: 2px solid var(--rw-teal);
  outline-offset: 2px; }}
</style>
"""


def finish(chart: alt.TopLevelMixin, height: int) -> alt.TopLevelMixin:
    """Restwave's chart look on a paper card: transparent, hairline grid, quiet axes.

    Vega measures axis labels short in this font, so the charts reserve their y-axis width
    (``minExtent``) and keep a little padding on both sides: otherwise a phone cuts labels off.
    """
    font = BODY.split(",")[0].strip('"')
    return (
        chart.properties(
            height=height, autosize=alt.AutoSizeParams(type="fit-x", contains="padding")
        )
        .configure(
            background="transparent",
            font=font,
            padding={"left": 6, "right": 14, "top": 6, "bottom": 2},
        )
        .configure_view(stroke=None)
        .configure_axis(
            domainColor=AXIS,
            gridColor=GRID,
            tickColor=AXIS,
            labelColor=TEXT_2,
            titleColor=TEXT_3,
            labelFontSize=12,
            titleFontSize=12,
            titleFontWeight=500,
            titlePadding=10,
            gridWidth=1,
            domainWidth=1,
            tickSize=4,
            labelFont=font,
            titleFont=font,
        )
    )
