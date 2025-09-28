import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

here = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(here, "results.csv")
df = pd.read_csv(csv_path)

LABEL_COL = "File"
VALUE_COL = "CER"

if LABEL_COL not in df.columns or VALUE_COL not in df.columns:
    raise ValueError(f"CSV must have columns '{LABEL_COL}' and '{VALUE_COL}'. "
                     f"Found: {df.columns.tolist()}")

df[VALUE_COL] = df[VALUE_COL].astype(float) * 100.0

df = df.reset_index(drop=True)
df["y"] = np.arange(len(df)) 

q1, q3 = df[VALUE_COL].quantile([0.25, 0.75])
iqr = q3 - q1
upper_fence = q3 + 1.5 * iqr

mask_out = df[VALUE_COL] > upper_fence
non_out = df.loc[~mask_out].copy()
out     = df.loc[ mask_out].copy()

if out.empty and df[VALUE_COL].max() > 5 * max(1e-9, df[VALUE_COL].median()):
    split_val = df[VALUE_COL].median() * 5
    mask_out = df[VALUE_COL] > split_val
    non_out = df.loc[~mask_out].copy()
    out     = df.loc[ mask_out].copy()

def tick_pct(x, _pos):
    return f"{x:.0f}%"

def label_pct(v):
    if v < 10:   return f"{v:.2f}%"
    if v < 100:  return f"{v:.1f}%"
    return f"{v:.0f}%"

if out.empty:
    fig_h = max(4.0, 0.5 * len(df))
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(df["y"], df[VALUE_COL])
    ax.set_yticks(df["y"])
    ax.set_yticklabels(df[LABEL_COL])
    ax.xaxis.set_major_formatter(FuncFormatter(tick_pct))
    fig.suptitle("Character Error Rate (CER) per file")
    fig.supxlabel("CER (%)")
    ax.grid(axis='x', linestyle=':', linewidth=0.6)
    for y, v in zip(df["y"], df[VALUE_COL]):
        ax.text(v, y, f" {label_pct(v)}", va="center", ha="left")
    plt.tight_layout()
    out_png = os.path.join(here, "cer_broken_axis.png")
    out_pdf = os.path.join(here, "cer_broken_axis.pdf")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_png}\nSaved: {out_pdf}")
    raise SystemExit

left_max  = max(non_out[VALUE_COL].max() * 1.1, df[VALUE_COL].quantile(0.7))

right_min = out[VALUE_COL].min() * 0.80
right_max = df[VALUE_COL].max() * 1.05

if right_min <= left_max:
    right_min = left_max * 0.9


fig_h = max(4.8, 0.52 * len(df))
fig, (axL, axR) = plt.subplots(
    1, 2, sharey=True, figsize=(11.5, fig_h),
    gridspec_kw={"width_ratios": [3.2, 1.2]}
)


axL.barh(non_out["y"], non_out[VALUE_COL])

for y, v in zip(out["y"], out[VALUE_COL]):
    axL.barh(y, left_max, color="C0")

axR.barh(out["y"], out[VALUE_COL], color="C0")

axL.set_xlim(0, left_max)
axR.set_xlim(right_min, right_max)

axL.set_yticks(df["y"])
axL.set_yticklabels(df[LABEL_COL], fontsize=9)

axL.yaxis.set_ticks_position("left")
axL.yaxis.set_label_position("left")
axL.spines["left"].set_position(("outward", 0))

axR.tick_params(axis="y", which="both", left=False, labelleft=False)
axR.spines["left"].set_visible(False)


for ax in (axL, axR):
    ax.xaxis.set_major_formatter(FuncFormatter(tick_pct))
    ax.grid(axis='x', linestyle=':', linewidth=0.6)

fig.supxlabel("CER (%)")
fig.suptitle("Character Error Rate (CER) per file")

axL.spines['right'].set_visible(False)
axR.spines['left'].set_visible(False)
d = 0.012
kwargs = dict(color='k', clip_on=False, linewidth=1)
axL.plot((1 - d, 1 + d), (-d, +d), transform=axL.transAxes, **kwargs)
axL.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=axL.transAxes, **kwargs)
axR.plot((-d, +d), (-d, +d), transform=axR.transAxes, **kwargs)
axR.plot((-d, +d), (1 - d, 1 + d), transform=axR.transAxes, **kwargs)

for y, v in zip(non_out["y"], non_out[VALUE_COL]):
    axL.text(v, y, f" {label_pct(v)}", va="center", ha="left")
for y, v in zip(out["y"], out[VALUE_COL]):
    axR.text(v, y, f" {label_pct(v)}", va="center", ha="left")

fig.subplots_adjust(wspace=0.06)
plt.tight_layout(rect=[0, 0, 1, 0.95])

out_png = os.path.join(here, "cer_broken_axis.png")
out_pdf = os.path.join(here, "cer_broken_axis.pdf")
plt.savefig(out_png, dpi=300, bbox_inches="tight")
plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
print(f"Saved: {out_png}\nSaved: {out_pdf}")
