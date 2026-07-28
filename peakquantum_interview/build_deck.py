#!/usr/bin/env python3
import os
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml
from PIL import Image

ROOT = "/Users/nikolaygusarov/plasmon/peakquantum_interview"
FIN = f"{ROOT}/assets/final"
MED = f"{ROOT}/assets/plasmon_media"
FMED = f"{ROOT}/assets/flux_media"

# ---------- palette ----------
BG      = RGBColor(0x0C, 0x14, 0x22)
CARD    = RGBColor(0x16, 0x22, 0x39)
CARD2   = RGBColor(0x1B, 0x29, 0x43)
LINE    = RGBColor(0x2C, 0x3E, 0x5C)
COPPER  = RGBColor(0xE0, 0x8A, 0x4C)
COPPERL = RGBColor(0xF4, 0xB0, 0x72)
TEAL    = RGBColor(0x54, 0xD8, 0xC4)
TEXT    = RGBColor(0xEE, 0xF3, 0xF8)
MUTED   = RGBColor(0x9F, 0xB4, 0xCC)
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
PILL    = RGBColor(0x20, 0x2F, 0x4A)
FONT = "Calibri"

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
BLANK = prs.slide_layouts[6]

def slide_bg(s):
    s.shapes.add_picture(f"{FIN}/bg.png", 0, 0, Inches(13.333), Inches(7.5))

def shadow(shape, blur=6, dist=3, dir_deg=90, alpha=0.42):
    spPr = shape._element.spPr
    xml = (f'<a:effectLst xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
           f'<a:outerShdw blurRad="{int(blur*12700)}" dist="{int(dist*12700)}" '
           f'dir="{int(dir_deg*60000)}" rotWithShape="0">'
           f'<a:srgbClr val="000000"><a:alpha val="{int(alpha*100000)}"/></a:srgbClr>'
           f'</a:outerShdw></a:effectLst>')
    spPr.append(parse_xml(xml))

def panel(s, x, y, w, h, fill=CARD, radius=0.055, line=LINE, lw=0.75, sh=True):
    sp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.adjustments[0] = radius
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None:
        sp.line.fill.background()
    else:
        sp.line.color.rgb = line; sp.line.width = Pt(lw)
    sp.shadow.inherit = False
    if sh: shadow(sp)
    return sp

def rrect(s, x, y, w, h, fill, radius=0.5, line=None, lw=1.0):
    sp = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.adjustments[0] = radius
    sp.fill.solid(); sp.fill.fore_color.rgb = fill
    if line is None: sp.line.fill.background()
    else: sp.line.color.rgb = line; sp.line.width = Pt(lw)
    sp.shadow.inherit = False
    return sp

def text(s, runs, x, y, w, h, size=14, color=TEXT, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, italic=False, spacing=1.0):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    tf.vertical_anchor = anchor
    if isinstance(runs, str):
        runs = [(runs, {})]
    p = tf.paragraphs[0]; p.alignment = align
    if spacing != 1.0: p.line_spacing = spacing
    for txt, o in runs:
        r = p.add_run(); r.text = txt
        r.font.name = FONT
        r.font.size = Pt(o.get("size", size))
        r.font.bold = o.get("bold", bold)
        r.font.italic = o.get("italic", italic)
        r.font.color.rgb = o.get("color", color)
    return tb

def fit(s, path, bx, by, bw, bh, halign="center", valign="middle"):
    iw, ih = Image.open(path).size
    ar = iw / ih
    if ar > bw / bh:
        w = bw; h = bw / ar
    else:
        h = bh; w = bh * ar
    x = bx + (bw - w) * (0.5 if halign == "center" else (0.0 if halign == "left" else 1.0))
    y = by + (bh - h) * (0.5 if valign == "middle" else (0.0 if valign == "top" else 1.0))
    return s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))

def thumb(s, path, bx, by, bw, bh, frame=None, pad=0.06):
    """frame='white' -> put a white rounded card behind (for plots on white bg)."""
    if frame == "white":
        iw, ih = Image.open(path).size
        ar = iw / ih
        if ar > (bw - 2 * pad) / (bh - 2 * pad):
            w = bw - 2 * pad; h = w / ar
        else:
            h = bh - 2 * pad; w = h * ar
        fx = bx + (bw - w) / 2; fy = by + (bh - h) / 2
        rrect(s, fx - pad, fy - pad, w + 2 * pad, h + 2 * pad, WHITE, radius=0.05)
        return s.shapes.add_picture(path, Inches(fx), Inches(fy), Inches(w), Inches(h))
    else:
        return fit(s, path, bx, by, bw, bh)

def pill(s, txt, x, y, w, h, fill=PILL, color=TEXT, size=12, bold=True, line=None):
    rrect(s, x, y, w, h, fill, radius=0.5, line=line, lw=1.0)
    text(s, [(txt, {"size": size, "bold": bold, "color": color})],
         x, y, w, h, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)

def chevron(s, x, y, w=0.30, h=0.42, color=COPPER):
    sp = s.shapes.add_shape(MSO_SHAPE.CHEVRON, Inches(x), Inches(y), Inches(w), Inches(h))
    sp.fill.solid(); sp.fill.fore_color.rgb = color
    sp.line.fill.background(); sp.shadow.inherit = False
    return sp

def header_dot(s, txt, x, y, w, color=COPPER, size=15):
    # small copper square marker + bold header text
    m = 0.12
    sq = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x), Inches(y + 0.045), Inches(m), Inches(m))
    sq.fill.solid(); sq.fill.fore_color.rgb = color; sq.line.fill.background(); sq.shadow.inherit = False
    text(s, [(txt, {"size": size, "bold": True, "color": TEXT})],
         x + m + 0.10, y, w - m - 0.10, 0.32, anchor=MSO_ANCHOR.MIDDLE)

def caption(s, txt, x, y, w, h=0.5, color=MUTED, size=10, align=PP_ALIGN.CENTER):
    text(s, [(txt, {"size": size, "color": color})], x, y, w, h, align=align,
         anchor=MSO_ANCHOR.MIDDLE, spacing=1.0)

# =====================================================================
# SLIDE 1 — Title + pipeline
# =====================================================================
s = prs.slides.add_slide(BLANK); slide_bg(s)

text(s, [("Full-Stack Superconducting-Qubit Design", {"size": 31, "bold": True, "color": TEXT})],
     0.6, 0.42, 12.2, 0.7)
text(s, [("One pipeline — from electromagnetic simulation to fabricated, packaged, measured devices",
          {"size": 14.5, "color": COPPERL})], 0.62, 1.12, 12, 0.4)
text(s, [("Nikolai Gusarov   ·   MSc Quantum Science & Engineering, EPFL   ·   Manucharyan Lab",
          {"size": 11.5, "color": MUTED})], 0.62, 1.5, 12, 0.35)

stages = [
    ("1", "Ansys HFSS", "3D electromagnetic\nsimulation", f"{MED}/image11.png", "white"),
    ("2", "Freqs + Capacitances", "mode ωₙ, Q-factors,\nfull C-matrix", f"{FIN}/cap_arrays.png", "white"),
    ("3", "Proprietary code", "Hamiltonian · κ · g\nPurcell T₁ · Tφ", f"{FIN}/mode_freqs.png", "white"),
    ("4", "gdsfactory layout", "parametric GDS\nNb (optical) + Al (e-beam)", f"{FIN}/gds_chip.png", None),
    ("5", "Fab + cryo measurement", "wafer → mounted chip\n→ dilution fridge", f"{MED}/image54.jpg", None),
]
cw, gap = 2.22, 0.31
x0, cy, ch = 0.49, 2.12, 3.98
imh = 1.98
for i, (num, lab, note, img, fr) in enumerate(stages):
    x = x0 + i * (cw + gap)
    panel(s, x, cy, cw, ch)
    # number badge
    b = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(x + 0.13), Inches(cy + 0.13), Inches(0.34), Inches(0.34))
    b.fill.solid(); b.fill.fore_color.rgb = COPPER; b.line.fill.background(); b.shadow.inherit = False
    text(s, [(num, {"size": 13, "bold": True, "color": BG})], x + 0.13, cy + 0.135, 0.34, 0.34,
         align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE)
    thumb(s, img, x + 0.16, cy + 0.16, cw - 0.32, imh, frame=fr)
    text(s, [(lab, {"size": 12.5, "bold": True, "color": COPPERL})],
         x + 0.12, cy + imh + 0.30, cw - 0.24, 0.6, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.TOP)
    text(s, [(note, {"size": 9.8, "color": MUTED})],
         x + 0.12, cy + imh + 0.92, cw - 0.24, 0.95, align=PP_ALIGN.CENTER, spacing=1.05)
    if i < len(stages) - 1:
        chevron(s, x + cw + (gap - 0.30) / 2, cy + 0.16 + imh / 2 - 0.21)

pill(s, "Closed design loop  —  the same pipeline drives both the Plasmonium and Fluxonium projects",
     2.06, 6.42, 9.2, 0.52, fill=PILL, color=TEXT, size=12.5, line=LINE)

s.notes_slide.notes_text_frame.text = (
    "Framing slide. My work is end-to-end quantum-processor design. Everything runs through one pipeline: "
    "(1) Ansys HFSS for 3D EM simulation; (2) extract mode frequencies, Q-factors and the full capacitance "
    "matrix; (3) feed them into my own Python code that builds the circuit Hamiltonian and computes kappa, "
    "coupling g, Purcell T1 and charge-noise T-phi; (4) turn the chosen parameters into a parametric GDS in "
    "gdsfactory with two lithography layers — Nb by optical litho, Al junctions by e-beam; (5) fabricate, "
    "package, and measure in the dilution fridge, then feed results back. Same pipeline for both projects.")

# =====================================================================
# SLIDE 2 — Plasmonium design showcase
# =====================================================================
s = prs.slides.add_slide(BLANK); slide_bg(s)
text(s, [("Plasmonium — designing a high-impedance Josephson-chain environment",
          {"size": 24, "bold": True, "color": TEXT})], 0.55, 0.34, 12.2, 0.55)
text(s, [("MSc thesis · circuit → EM → layout → package, taken all the way to a mounted chip",
          {"size": 12.5, "color": COPPERL})], 0.57, 0.92, 12, 0.35)

# quadrant panels
PX = [0.5, 6.83]
PY = [1.42, 4.02]
PW, PH = 6.0, 2.5

def quad(col, row):
    return PX[col], PY[row], PW, PH

# Q1 circuit & physics
x, y, w, h = quad(0, 0); panel(s, x, y, w, h)
header_dot(s, "Circuit & physics", x + 0.18, y + 0.14, w - 0.36)
thumb(s, f"{FIN}/circuit.png", x + 0.18, y + 0.52, w - 0.36, h - 1.02, frame="white")
caption(s, "High-impedance JJ chain (N ≈ 50–100) + SQUID, coupled to a 50 Ω feedline via Cᴄ",
        x + 0.2, y + h - 0.46, w - 0.4)

# Q2 EM design (HFSS)
x, y, w, h = quad(1, 0); panel(s, x, y, w, h)
header_dot(s, "EM design — Ansys HFSS", x + 0.18, y + 0.14, w - 0.36)
iw = (w - 0.36 - 2 * 0.12) / 3
thumb(s, f"{MED}/image11.png", x + 0.18, y + 0.52, iw, h - 1.02, frame="white")
thumb(s, f"{MED}/image35.png", x + 0.18 + iw + 0.12, y + 0.52, iw, h - 1.02, frame="white")
thumb(s, f"{MED}/image37.png", x + 0.18 + 2 * (iw + 0.12), y + 0.52, iw, h - 1.02, frame="white")
caption(s, "3D model + mode E-fields — chain modes 20–25 GHz, qubit mode ~5 GHz",
        x + 0.2, y + h - 0.46, w - 0.4)

# Q3 layout (gdsfactory)
x, y, w, h = quad(0, 1); panel(s, x, y, w, h)
header_dot(s, "Layout — gdsfactory (GDS)", x + 0.18, y + 0.14, w - 0.36)
thumb(s, f"{FIN}/gds_wafer.png", x + 0.18, y + 0.5, 2.35, h - 1.0)
thumb(s, f"{FIN}/gds_chip.png", x + 0.18 + 2.35 + 0.1, y + 0.5, 2.35, h - 1.0)
thumb(s, f"{FIN}/gds_filter.png", x + 0.18 + 2 * 2.35 + 0.2, y + 0.5, 0.72, h - 1.0)
caption(s, "100 mm wafer · 6 qubits/chip · 6.37 GHz Purcell filters   (Nb optical + Al e-beam)",
        x + 0.2, y + h - 0.46, w - 0.4)

# Q4 package & fab
x, y, w, h = quad(1, 1); panel(s, x, y, w, h)
header_dot(s, "Package & fabrication", x + 0.18, y + 0.14, w - 0.36)
iw2 = (w - 0.36 - 2 * 0.12) / 3
thumb(s, f"{FIN}/pkg_cad.png", x + 0.18, y + 0.52, iw2, h - 1.02)
thumb(s, f"{MED}/image55.png", x + 0.18 + iw2 + 0.12, y + 0.52, iw2, h - 1.02, frame="white")
thumb(s, f"{MED}/image54.jpg", x + 0.18 + 2 * (iw2 + 0.12), y + 0.52, iw2, h - 1.02)
caption(s, "Sample package — Inventor CAD + HFSS   ·   fabricated & mounted in copper holder",
        x + 0.2, y + h - 0.46, w - 0.4)

# results strip
ry = 6.68
res = ["κ = 5–10 MHz", "ω₁ = 20–25 GHz", "g = 100–500 MHz", "→ ready to fabricate"]
rw = 2.7; rg = 0.28; tot = 4 * rw + 3 * rg; sx = (13.333 - tot) / 2
for i, r in enumerate(res):
    fill = COPPER if i == 3 else PILL
    col = BG if i == 3 else TEXT
    pill(s, r, sx + i * (rw + rg), ry, rw, 0.5, fill=fill, color=col, size=13, line=(None if i == 3 else LINE))

s.notes_slide.notes_text_frame.text = (
    "Plasmonium is my flagship design project. Top-left: the circuit — a high-impedance Josephson-junction "
    "chain (50-100 junctions) with a SQUID, capacitively coupled to a 50-ohm feedline; this is the "
    "high-impedance environment. Top-right: I model it in HFSS and pull out the mode E-fields — chain modes "
    "at 20-25 GHz and the qubit near 5 GHz. Bottom-left: the actual gdsfactory layout — a full 100 mm wafer, "
    "six qubits per chip, 6.37 GHz Purcell meander filters, drawn in two litho layers (Nb optical, Al "
    "e-beam junctions). Bottom-right: I also designed the microwave sample package in Inventor, simulated it "
    "in HFSS, and here it is fabricated and mounted. Bottom line: I identified designs hitting kappa 5-10 MHz, "
    "first chain mode 20-25 GHz, g 100-500 MHz — ready to fab.")

# =====================================================================
# SLIDE 3 — From design to data
# =====================================================================
s = prs.slides.add_slide(BLANK); slide_bg(s)
text(s, [("From design to data — measured devices & readout engineering",
          {"size": 24, "bold": True, "color": TEXT})], 0.55, 0.34, 12.2, 0.55)
text(s, [("Closing the loop: designs become cooled-down qubits, characterised and fed back",
          {"size": 12.5, "color": COPPERL})], 0.57, 0.92, 12, 0.35)

# left panel — Plasmonium first cooldown
lx, ly, lw, lh = 0.5, 1.42, 5.0, 4.7
panel(s, lx, ly, lw, lh)
header_dot(s, "Plasmonium — first cooldown", lx + 0.2, ly + 0.16, lw - 0.4)
thumb(s, f"{FIN}/pl_ro_power.png", lx + 0.22, ly + 0.62, lw - 0.44, 2.0, frame="white")
thumb(s, f"{MED}/image47.png", lx + 0.22, ly + 2.72, lw - 0.44, 1.15, frame="white")
caption(s, "Resonator spectroscopy vs power & flux/current — modes appear and shift exactly as designed",
        lx + 0.22, ly + 3.95, lw - 0.44, 0.6)

# right panel — Fluxonium characterisation
rx, ry2, rw2, rh = 5.72, 1.42, 7.1, 4.7
panel(s, rx, ry2, rw2, rh)
header_dot(s, "Fluxonium — coherence & readout characterisation", rx + 0.2, ry2 + 0.16, rw2 - 0.4)
tw = (rw2 - 0.44 - 2 * 0.14) / 3
thumb(s, f"{FIN}/flux_anticross.png", rx + 0.22, ry2 + 0.62, tw, 2.0, frame="white")
thumb(s, f"{FIN}/flux_singleshot.png", rx + 0.22 + tw + 0.14, ry2 + 0.62, tw, 2.0, frame="white")
thumb(s, f"{FIN}/flux_rb.png", rx + 0.22 + 2 * (tw + 0.14), ry2 + 0.62, tw, 2.0, frame="white")
c1 = rx + 0.22 + tw / 2
c2 = rx + 0.22 + tw + 0.14 + tw / 2
c3 = rx + 0.22 + 2 * (tw + 0.14) + tw / 2
caption(s, "qubit–resonator anti-crossing", c1 - tw / 2, ry2 + 2.66, tw, 0.35, size=9.5)
caption(s, "single-shot readout (IQ)", c2 - tw / 2, ry2 + 2.66, tw, 0.35, size=9.5)
caption(s, "randomised benchmarking", c3 - tw / 2, ry2 + 2.66, tw, 0.35, size=9.5)
caption(s, "Readout tuning, single-shot, reset, T₁ / T₂, flux- vs charge-drive gates",
        rx + 0.22, ry2 + 3.2, rw2 - 0.44, 0.4, size=10.5, color=TEXT)
# RED highlight inside right panel
rrect(s, rx + 0.22, ry2 + 3.72, rw2 - 0.44, 0.72, CARD2, radius=0.08, line=COPPER, lw=1.0)
text(s, [("Insight — RED (Readout Error Decoder): ", {"size": 11, "bold": True, "color": COPPERL}),
         ("repeated single-shot records → a 4-channel error budget (assignment · Pauli · short-/long-lived leakage)",
          {"size": 11, "color": TEXT})],
     rx + 0.42, ry2 + 3.72, rw2 - 0.84, 0.72, anchor=MSO_ANCHOR.MIDDLE, spacing=1.05)

# footer skills strip
sy = 6.42
skills = ["HFSS / EM", "Circuit Hamiltonians", "gdsfactory GDS", "Inventor CAD", "Cryogenic RF measurement"]
sw = 2.28; sg = 0.2; tot = len(skills) * sw + (len(skills) - 1) * sg; sx = (13.333 - tot) / 2
for i, sk in enumerate(skills):
    pill(s, sk, sx + i * (sw + sg), sy, sw, 0.5, fill=PILL, color=TEAL, size=11.5, line=LINE)

s.notes_slide.notes_text_frame.text = (
    "Design only matters if the devices work, so I also measure and characterise. Left: the first plasmonium "
    "cooldown — resonator spectroscopy vs power and vs flux/current; the modes appear and move exactly as the "
    "design predicts. Right: on fluxonium I did the full characterisation stack — qubit-resonator "
    "anti-crossings, single-shot readout optimisation, randomised benchmarking, reset, T1/T2, and flux- vs "
    "charge-drive gates. The key insight, RED: from plain repeated single-shot records plus known gates you "
    "can reconstruct a four-channel readout-error budget — assignment, Pauli, and short- and long-lived "
    "leakage — which ordinary assignment-fidelity metrics hide. So I cover the whole loop: design, fabricate, "
    "measure, and feed the physics back into the next design.")

out = f"{ROOT}/Gusarov_PeakQuantum_QubitDesign.pptx"
prs.save(out)
print("saved", out)
