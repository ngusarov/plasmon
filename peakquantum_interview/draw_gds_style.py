#!/usr/bin/env python3
"""Two-qubit concept layouts drawn in KLayout/GDS style (hatched layers, junction marks)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, FancyBboxPatch
plt.rcParams["hatch.linewidth"] = 0.7
plt.rcParams["font.family"] = "DejaVu Sans"

OUT = "/Users/nikolaygusarov/plasmon/peakquantum_interview/assets/final"

TEAL  = "#1D9C9C"   # JJ chain (Al)
GREEN = "#3AAE3A"   # SQUID (Al)
OLIVE = "#9A8B3C"   # Nb pad / capacitor
PGRN  = "#AFD08A"   # ground
BLUE  = "#2438C6"   # fluxonium metal (Nb)
ORANGE= "#E07A12"   # flux line
RED   = "#E23B3B"
JGRN  = "#22C022"
TXT   = "#2B2B2B"
GRID  = "#D4D4D4"

def setup(xlim, ylim, w=11, h=7):
    fig = plt.figure(figsize=(w, h), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    for X in range(int(xlim[0]), int(xlim[1]) + 1, 8):
        for Y in range(int(ylim[0]), int(ylim[1]) + 1, 8):
            ax.plot(X, Y, ".", color=GRID, ms=1.5, zorder=0)
    return fig, ax

def rect(ax, x, y, w, h, color, hatch="////", lw=1.3, fc="none", z=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=color, hatch=hatch, linewidth=lw, zorder=z))

def rrect(ax, x, y, w, h, color, hatch="////", lw=1.3, r=2.2, z=2):
    ax.add_patch(FancyBboxPatch((x + r, y + r), w - 2 * r, h - 2 * r,
                 boxstyle=f"round,pad={r},rounding_size={r}", facecolor="none",
                 edgecolor=color, hatch=hatch, linewidth=lw, zorder=z))

def jmark(ax, x, y, s=1.1):
    ax.add_patch(Rectangle((x - s, y - s * 0.5), s * 1.7, s, facecolor=RED, edgecolor="none", zorder=6))
    ax.add_patch(Rectangle((x - s * 0.4, y - s), s, s * 1.7, facecolor=JGRN, edgecolor="none", zorder=6))

def squid(ax, cx, cy, w=12, h=10, color=GREEN):
    t = w * 0.26
    rect(ax, cx - w / 2, cy - h / 2, w, h, color, hatch="\\\\\\\\")
    ax.add_patch(Rectangle((cx - w / 2 + t, cy - h / 2 + t), w - 2 * t, h - 2 * t,
                 facecolor="white", edgecolor=color, linewidth=1.0, zorder=3))
    jmark(ax, cx - w / 2 + t / 2, cy); jmark(ax, cx + w / 2 - t / 2, cy)

def bowtie(ax, cx, cy, w=13, h=12, color=BLUE):
    g = 1.0
    up = [(cx - w / 2, cy + h / 2), (cx + w / 2, cy + h / 2), (cx + g, cy + g), (cx - g, cy + g)]
    dn = [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + g, cy - g), (cx - g, cy - g)]
    ax.add_patch(Polygon(up, closed=True, facecolor="none", edgecolor=color, hatch="////", linewidth=1.3, zorder=2))
    ax.add_patch(Polygon(dn, closed=True, facecolor="none", edgecolor=color, hatch="////", linewidth=1.3, zorder=2))
    jmark(ax, cx, cy)

def chain(ax, x, y, length, width=3.2, horizontal=False, color=TEAL, n=None):
    if horizontal:
        rect(ax, x, y - width / 2, length, width, color, hatch="////")
        if n:
            for i in range(1, n):
                xx = x + length * i / n
                ax.plot([xx, xx], [y - width / 2, y + width / 2], color=color, lw=0.6, zorder=3)
    else:
        rect(ax, x - width / 2, y, width, length, color, hatch="////")
        if n:
            for i in range(1, n):
                yy = y + length * i / n
                ax.plot([x - width / 2, x + width / 2], [yy, yy], color=color, lw=0.6, zorder=3)

def jarray(ax, x0, y0, x1, y1, n=12, box=2.6, color=TEAL):
    ax.plot([x0, x1], [y0, y1], color=color, lw=1.0, zorder=2)
    for i in range(n):
        t = (i + 0.5) / n
        rect(ax, x0 + (x1 - x0) * t - box / 2, y0 + (y1 - y0) * t - box / 2, box, box, color, hatch="////", lw=0.9)

def idc(ax, x0, x1, yc, height, n=5, color=OLIVE, fw=1.4):
    gap = x1 - x0
    sL = x0 + gap * 0.14; sR = x1 - gap * 0.14
    rect(ax, sL - fw, yc - height / 2, fw * 2, height, color, hatch="", lw=1.1, fc=color)
    rect(ax, sR - fw, yc - height / 2, fw * 2, height, color, hatch="", lw=1.1, fc=color)
    fl = (sR - sL) * 0.74
    for i in range(n):
        yy = yc - height / 2 + height * (i + 0.5) / n
        if i % 2 == 0:
            rect(ax, sL, yy - fw / 2, fl, fw, color, hatch="", lw=0.9, fc=color)
        else:
            rect(ax, sR - fl, yy - fw / 2, fl, fw, color, hatch="", lw=0.9, fc=color)

def fluxline(ax, x, y0, y1, w=1.7):
    rect(ax, x - w / 2, min(y0, y1), w, abs(y1 - y0), ORANGE, hatch="", lw=1.0, fc=ORANGE)

def lab(ax, text, tx, ty, px=None, py=None, color=TXT, size=12, ha="center", va="center", bold=False):
    if px is not None:
        ax.annotate(text, xy=(px, py), xytext=(tx, ty), color=color, fontsize=size, ha=ha, va=va,
                    fontweight=("bold" if bold else "normal"),
                    arrowprops=dict(arrowstyle="-", color="#8A8A8A", lw=0.9, shrinkA=1, shrinkB=2), zorder=8)
    else:
        ax.text(tx, ty, text, color=color, fontsize=size, ha=ha, va=va,
                fontweight=("bold" if bold else "normal"), zorder=8)

# =====================================================================
# PL — Route A: two qubits + shared flux-tunable coupler SQUID
# =====================================================================
fig, ax = setup((0, 120), (0, 70), w=11.8, h=6.9)
rect(ax, 6, 3, 108, 3.5, PGRN, hatch="", lw=1.0)                 # ground rail

def merg(cx, pad_w=24):
    rect(ax, cx - pad_w / 2, 26, pad_w, 18, OLIVE, hatch="////")  # capacitor pad
    rect(ax, cx - 1.8, 20, 3.6, 6, TEAL, hatch="")              # pad -> SQUID
    squid(ax, cx, 15, w=12, h=10)                              # mergemon SQUID (y10..20)
    rect(ax, cx - 1.6, 6, 3.2, 4.5, TEAL, hatch="")            # SQUID -> ground
    chain(ax, cx, 44, 12, color=TEAL, n=4)                     # readout chain up

merg(24); merg(96)                                              # Qubit A, Qubit B

# tunable coupler
cx = 60
rect(ax, cx - 9, 28, 18, 14, OLIVE, hatch="////")              # coupler island
rect(ax, cx - 1.8, 22, 3.6, 6, TEAL, hatch="")                # island -> SQUID
squid(ax, cx, 17, w=11, h=9)                                  # y12.5..21.5
rect(ax, cx - 1.6, 6, 3.2, 6.5, TEAL, hatch="")              # SQUID -> ground
fluxline(ax, cx + 12, 0, 22)                                   # flux line

idc(ax, 36, 51, 35, 13, n=4)                                   # Cc  A<->coupler
idc(ax, 69, 84, 35, 13, n=4)                                   # Cc  coupler<->B

lab(ax, "Qubit A", 24, 64, bold=True, size=14)
lab(ax, "Qubit B", 96, 64, bold=True, size=14)
lab(ax, "tunable coupler SQUID  (+ Φ)", 60, 64, bold=True, size=12.5, color="#B06A16")
lab(ax, "SQUID\n(Al)", 10, 15, 18, 15, size=10.5, ha="center")
lab(ax, "capacitor pad (Nb)", 24, 52, 24, 44, size=10.5)
lab(ax, "readout chain →", 40, 55, 27, 52, size=10.5, ha="left")
lab(ax, "Cᴄ", 43.5, 44, size=11, color="#B06A16", bold=True)
lab(ax, "Cᴄ", 76.5, 44, size=11, color="#B06A16", bold=True)
lab(ax, "Φ", 72, 25, size=12, color=ORANGE, bold=True)
lab(ax, "ground", 60, 1.5, size=9.5, color="#6f8a4a")
fig.savefig(f"{OUT}/gdsstyle_pl_coupler.png", facecolor="white", dpi=200)
plt.close(fig); print("saved gdsstyle_pl_coupler.png")

# =====================================================================
# PL — Route B: two qubits sharing one high-impedance chain (quantum bus)
# =====================================================================
fig, ax = setup((0, 120), (0, 84), w=11.5, h=8.0)
BUSY = 52
rect(ax, 6, 74, 108, 4, OLIVE, hatch="////")                   # feedline
chain(ax, 26, BUSY + 2, 20, horizontal=False, color=TEAL)      # readout stub -> feedline
lab(ax, "feedline", 90, 80, size=12)
lab(ax, "Cᴄ", 30, 71, 27, 74, size=11, color="#B06A16", bold=True)

chain(ax, 20, BUSY, length=80, horizontal=True, width=3.8, n=18)   # shared chain (bus)
rect(ax, 10, BUSY - 6, 10, 12, OLIVE, hatch="////")            # end ground pads
rect(ax, 100, BUSY - 6, 10, 12, OLIVE, hatch="////")

def tap(xc, name):
    chain(ax, xc, 28, BUSY - 28, horizontal=False, color=TEAL)  # stub down from bus
    squid(ax, xc, 24, w=12, h=10)                              # mergemon SQUID
    rect(ax, xc - 11, 6, 22, 12, OLIVE, hatch="////")          # island / pad
    rect(ax, xc - 1.6, 18, 3.2, 2.5, TEAL, hatch="")
    lab(ax, name, xc, 40, bold=True, size=13)

tap(45, "Qubit A"); tap(78, "Qubit B")
lab(ax, "shared high-Z JJ chain  =  quantum bus", 60, 62, size=13, color="#0F7A7A", bold=True)
lab(ax, "SQUID qubit (Al)", 22, 24, 34, 24, size=10.5, ha="center")
lab(ax, "capacitor pad (Nb)", 78, 2, 78, 6, size=10.5)
fig.savefig(f"{OUT}/gdsstyle_pl_bus.png", facecolor="white", dpi=200)
plt.close(fig); print("saved gdsstyle_pl_bus.png")

# =====================================================================
# FL — two fluxoniums, capacitive coupling
# =====================================================================
fig, ax = setup((-10, 144), (0, 82), w=12.6, h=6.9)

def fluxonium(cx, pw=22, gap=16, ph=30):
    xl = cx - gap / 2 - pw; xr = cx + gap / 2
    rrect(ax, xl, 30, pw, ph, BLUE); rrect(ax, xr, 30, pw, ph, BLUE)   # shunt pads
    my = 30 + ph / 2
    bowtie(ax, cx, my, w=gap - 2, h=13)                                # small JJ
    lx = xl + pw / 2; rx = xr + pw / 2
    ax.plot([lx, lx], [30 + ph, 30 + ph + 8], color=BLUE, lw=1.1, zorder=2)
    ax.plot([rx, rx], [30 + ph, 30 + ph + 8], color=BLUE, lw=1.1, zorder=2)
    jarray(ax, lx, 30 + ph + 8, rx, 30 + ph + 8, n=12, color=BLUE)     # superinductor
    fluxline(ax, cx, 4, 34)
    return xl, xr

aL, aR = fluxonium(30)          # A inner (right) pad ends at 30+8+22 = 60
bL, bR = fluxonium(110)         # B inner (left) pad starts at 110-8-22 = 80
idc(ax, 60, 80, 45, 26, n=6, color=BLUE)                              # coupling capacitor

lab(ax, "Fluxonium A", 30, 76, bold=True, size=14)
lab(ax, "Fluxonium B", 110, 76, bold=True, size=14)
lab(ax, "shunt capacitor\npads (Nb)", 0, 18, 8, 36, size=10.5, ha="center")
lab(ax, "small JJ  Eⱼ (Al)", 52, 20, 36, 45, size=10.5, ha="center")
lab(ax, "superinductor — JJ array (Al)", 70, 70, 49, 68, size=10.5, color="#1a1aa8")
lab(ax, "Cᴄ  coupling capacitor", 70, 24, 70, 35, size=12, color="#1a1aa8", bold=True)
lab(ax, "flux line  Φ", 30, 8, size=10, color=ORANGE)
lab(ax, "(tunable: add a coupler element in this gap)", 70, 15, size=9.5, color="#7a7a7a")
fig.savefig(f"{OUT}/gdsstyle_fl_cap.png", facecolor="white", dpi=200)
plt.close(fig); print("saved gdsstyle_fl_cap.png")
