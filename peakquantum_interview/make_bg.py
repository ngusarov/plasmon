import numpy as np
from PIL import Image

W, H = 2000, 1125
yy, xx = np.mgrid[0:H, 0:W].astype(float)
# vertical navy gradient, darker toward bottom
t = yy / H
top = np.array([18, 30, 52])      # 121E34
bot = np.array([8, 13, 23])       # 080D17
base = (top[None, None, :] * (1 - t)[..., None] + bot[None, None, :] * t[..., None])
# subtle copper glow top-left
cx, cy = 0.16 * W, 0.10 * H
r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (0.9 * W)
glow = np.clip(1 - r, 0, 1) ** 2
copper = np.array([70, 42, 22])
img = base + glow[..., None] * copper * 0.9
# faint vignette
r2 = np.sqrt((xx - 0.5 * W) ** 2 + (yy - 0.5 * H) ** 2) / (0.75 * W)
vig = np.clip(r2 - 0.55, 0, 1) * 0.5
img = img * (1 - vig[..., None] * 0.6)
img = np.clip(img, 0, 255).astype(np.uint8)
Image.fromarray(img).save("/Users/nikolaygusarov/plasmon/peakquantum_interview/assets/final/bg.png")
print("bg saved")
