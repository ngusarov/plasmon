#!/usr/bin/env python3
"""Concept two-qubit-coupling layouts, GDS/polygon style, Nb=copper Al=teal."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = "/Users/nikolaygusarov/plasmon/peakquantum_interview/assets/final"
COPPER = "#E3A45E"   # Nb (optical)
TEAL   = "#5FE0D0"   # Al / junctions (e-beam)
TXT    = "#EAF1F8"
ACC    = "#F4B778"

def new_ax(w=13.4, h=6.4):
    fig = plt.figure(figsize=(w, h), dpi=210)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(-8, 208); ax.set_ylim(-20, 116); ax.set_aspect("equal")
    return fig, ax

def pad(ax, x, y, w, h, color=COPPER, r=3.2):
    ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r,
                 boxstyle=f"round,pad={r},rounding_size={r}", fc=color, ec="none"))

def rect(ax, x, y, w, h, color, ec="none", lw=0):
    ax.add_patch(Rectangle((x, y), w, h, fc=color, ec=ec, lw=lw))

def wire(ax, x1, y1, x2, y2, w=1.6, color=COPPER):
    if abs(y1 - y2) < 1e-6:
        rect(ax, min(x1, x2), y1 - w / 2, abs(x2 - x1), w, color)
    elif abs(x1 - x2) < 1e-6:
        rect(ax, x1 - w / 2, min(y1, y2), w, abs(y2 - y1), color)
    else:
        ax.plot([x1, x2], [y1, y2], color=color, lw=w * 2.2, solid_capstyle="round")

def jj(ax, cx, cy, s=4.2, color=TEAL):
    rect(ax, cx - s / 2, cy - s / 2, s, s, color)

def array(ax, x0, y0, x1, y1, n=10, box=3.0, color=TEAL, w=1.3):
    wire(ax, x0, y0, x1, y1, w=w, color=color)
    for i in range(n):
        t = (i + 0.5) / n
        cx = x0 + (x1 - x0) * t; cy = y0 + (y1 - y0) * t
        rect(ax, cx - box / 2, cy - box / 2, box, box, color)

def squid(ax, cx, y_bot, y_top, w=11, color=TEAL, lw=3.2):
    # rectangular loop with two junctions on the vertical arms
    ax.add_patch(Rectangle((cx - w / 2, y_bot), w, y_top - y_bot, fc="none", ec=color, lw=lw))
    my = (y_bot + y_top) / 2
    jj(ax, cx - w / 2, my, s=5.0, color=color)
    jj(ax, cx + w / 2, my, s=5.0, color=color)

def idc(ax, x0, x1, yc, height=16, n=4, color=COPPER, fw=1.8):
    # interdigitated coupling capacitor between two combs
    gap = x1 - x0
    spineL = x0 + gap * 0.10; spineR = x1 - gap * 0.10
    rect(ax, spineL - fw, yc - height / 2, fw * 2, height, color)
    rect(ax, spineR - fw, yc - height / 2, fw * 2, height, color)
    fl = (spineR - spineL) * 0.80
    for i in range(n):
        yy = yc - height / 2 + height * (i + 0.5) / n
        if i % 2 == 0:
            rect(ax, spineL, yy - fw / 2, fl, fw, color)
        else:
            rect(ax, spineR - fl, yy - fw / 2, fl, fw, color)

def lab(ax, text, px, py, tx, ty, color=TXT, size=13, ha="center", va="center", arrow=True, bold=False):
    if arrow:
        ax.annotate(text, xy=(px, py), xytext=(tx, ty), color=color, fontsize=size,
                    ha=ha, va=va, fontweight=("bold" if bold else "normal"),
                    arrowprops=dict(arrowstyle="-", color="#8CA0B8", lw=1.1,
                                    shrinkA=2, shrinkB=3))
    else:
        ax.text(tx, ty, text, color=color, fontsize=size, ha=ha, va=va,
                fontweight=("bold" if bold else "normal"))

# =====================================================================
# DESIGN 1 — Plasmonium two-qubit via tunable coupler
# =====================================================================
fig, ax = new_ax()
GND_Y = 4
rect(ax, 12, GND_Y - 3, 176, 4, COPPER)          # ground rail
rect(ax, 8, 92, 184, 4.5, COPPER)                # readout feedline (top)

def plasq(cx, name_side):
    # qubit pad
    pad(ax, cx - 22, 42, 44, 30)
    # readout chain up to feedline (with small coupling gap)
    array(ax, cx, 72, cx, 88, n=6, box=3.2)
    # SQUID down to ground
    wire(ax, cx, 42, cx, 34, w=2.2, color=TEAL)
    squid(ax, cx, 20, 34)
    wire(ax, cx, 20, cx, GND_Y + 1, w=2.2, color=TEAL)

plasq(40, "L")
plasq(160, "R")

# tunable coupler (middle): pad + SQUID to ground + flux line
pad(ax, 78, 46, 44, 22)
wire(ax, 100, 46, 100, 40, w=2.2, color=TEAL)
squid(ax, 100, 26, 40, w=10)
wire(ax, 100, 26, 100, GND_Y + 1, w=2.2, color=TEAL)
# flux line to coupler loop (mutual)
wire(ax, 112, -14, 112, 30, w=1.8, color=COPPER)
wire(ax, 112, 30, 106, 33, w=1.8, color=COPPER)

# coupling capacitors (IDC) in the two gaps
idc(ax, 62, 78, 57, height=17, n=4)
idc(ax, 122, 138, 57, height=17, n=4)

# labels
lab(ax, "Qubit A", 40, 57, 40, 108, size=15, bold=True, arrow=False)
lab(ax, "Qubit B", 160, 57, 160, 108, size=15, bold=True, arrow=False)
lab(ax, "Tunable coupler", 100, 57, 100, 108, size=15, bold=True, color=ACC, arrow=False)
lab(ax, "mergemon:\ncapacitor pad + SQUID (Al)", 40, 27, 18, 74, size=11.5, ha="center")
lab(ax, "readout chain → feedline", 40, 84, 66, 100, size=11.5, ha="left")
lab(ax, "Cᴄ", 70, 57, 70, 68, size=13, color=ACC, arrow=False, bold=True)
lab(ax, "Cᴄ", 130, 57, 130, 68, size=13, color=ACC, arrow=False, bold=True)
lab(ax, "coupler SQUID\n+ flux line  Φ", 106, 30, 138, 20, size=11.5, ha="left")
lab(ax, "shared ground plane", 100, GND_Y - 2, 100, -12, size=11.5, arrow=False, color="#9FB4CC")
fig.savefig(f"{OUT}/twoqubit_plasmonium.png", transparent=True, dpi=210)
plt.close(fig); print("saved twoqubit_plasmonium.png")

# =====================================================================
# DESIGN 2 — Fluxonium two-qubit via capacitive coupling
# =====================================================================
fig, ax = new_ax()

def fluxonium(cx_in, outer_left):
    # two pads (shunt capacitor). outer_left=True -> outer pad on left
    if outer_left:
        px1, px2 = cx_in - 46, cx_in + 14      # outer(left), inner(right)
    else:
        px1, px2 = cx_in - 14, cx_in + 46      # inner(left), outer(right)
    pad(ax, px1, 34, 32, 34)
    pad(ax, px2, 34, 26, 34)
    gx0, gx1 = px1 + 32, px2                    # gap between pads
    my = 51
    # small JJ across the gap
    wire(ax, gx0, my, (gx0 + gx1) / 2 - 3, my, w=1.8, color=TEAL)
    wire(ax, (gx0 + gx1) / 2 + 3, my, gx1, my, w=1.8, color=TEAL)
    jj(ax, (gx0 + gx1) / 2, my, s=6, color=TEAL)
    # superinductor array (loop) over the top
    lx = px1 + 8; rx = px2 + 13
    wire(ax, lx, 68, lx, 80, w=1.4, color=TEAL)
    wire(ax, rx, 68, rx, 80, w=1.4, color=TEAL)
    array(ax, lx, 80, rx, 80, n=11, box=3.0)
    # flux line threading loop
    fxx = (gx0 + gx1) / 2
    wire(ax, fxx, -14, fxx, 44, w=1.7, color=COPPER)
    return px1, px2, gx0, gx1

aL = fluxonium(52, outer_left=True)
bR = fluxonium(148, outer_left=False)

# coupling capacitor between the two inner pads
idc(ax, aL[1] + 26, bR[0], 51, height=20, n=5)

# labels
lab(ax, "Fluxonium A", 52, 60, 45, 108, size=15, bold=True, arrow=False)
lab(ax, "Fluxonium B", 148, 60, 155, 108, size=15, bold=True, arrow=False)
lab(ax, "shunt capacitor\npads (Nb)", 20, 51, 8, 84, size=11.5, ha="center")
lab(ax, "small JJ  $E_J$ (Al)", 63, 51, 40, 24, size=12, ha="center")
lab(ax, "superinductor — JJ array (Al)", 66, 80, 96, 96, size=11.5, ha="center", color=TEAL)
lab(ax, "Cᴄ  capacitive coupling", 100, 51, 100, 22, size=12.5, color=ACC, bold=True)
lab(ax, "flux line  Φ", 66, 8, 92, -12, size=11.5, ha="center")
fig.savefig(f"{OUT}/twoqubit_fluxonium.png", transparent=True, dpi=210)
plt.close(fig); print("saved twoqubit_fluxonium.png")
