from __future__ import annotations
import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from dd2.utilities import z_pval
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from mfr.colors import colors


def visualize(decoders, keys, errorsbars=False, errorplot=True):
    num_decoders = len(decoders)
    num_keys = len(keys)

    COLORS = (
        colors.blue,
        colors.red,
        colors.green,
        colors.purple,
        colors.orange,
        colors.yellow
    )
    f = plt.figure(figsize=(2 + num_keys, 4))
    gs = f.add_gridspec(2, 1)
    a = f.add_subplot(gs[:, :])
    a.set_title("Decoding Performance")
    a.set_ylabel("Balanced Accuracy (%)")
    a.axhline(50, linestyle='--', color='k', lw=2)
    a.set_xlim([-1, num_keys])
    a.set_ylim([0, 100])
    a.set_xticks(range(-1, num_keys + 1, 1))
    labels = [key.upper() for key in keys]
    labels.insert(0, "")
    labels.append("")
    a.set_xticklabels(labels)
    a.tick_params(axis="y", which="minor", direction="in", length=3, width=1.5)
    a.tick_params(axis="y", which="major", direction="in", length=5, width=1.5)
    a.tick_params(axis="x", which="both", length=0, width=0)
    a.yaxis.set_major_locator(MultipleLocator(25))
    a.yaxis.set_minor_locator(MultipleLocator(5))

    for idx, decoder, key in zip(range(num_decoders), decoders, keys):

        real = decoder.real_performance.get(key)
        null = decoder.null_performance.get(key)

        a.scatter([idx], [real * 100], s=100, marker="d", color=COLORS[idx], facecolor="white",
                  linewidth=2)
        up = np.nanmean(null) * 100 + np.nanstd(null) * 100
        low = np.nanmean(null) * 100 - np.nanstd(null) * 100
        # a.fill([idx - 1, idx, idx + 1, idx + 1, idx, idx - 1], [up, up, up, low, low, low], alpha=0.3, color=(0.65, 0.65, 0.65))
        if errorsbars:
            a.errorbar([idx], np.nanmean(null) * 100, yerr=100 * np.nanstd(null), color="k",
                       linewidth=2, capsize=5, marker="_", alpha=0.95)

        p = z_pval(real, null)[-1]

        if p <= 0.05:
            if 0.01 < p <= 0.05:
                ast = "*"
            elif 0.001  < p <= 0.01:
                ast = "**"
            elif 0.0001 < p <= 0.001:
                ast = "***"
            else:
                ast = "****"

            ctr = np.nanmean(null) + real
            ctr /= 2
            ctr *= 100
            diff = np.nanmean(null) - real
            diff /= 4
            diff *= 100
            # sig
            a.plot([-0.2 + idx, -0.2 + idx],
                   [ctr - diff, ctr + diff],
                   color="k", alpha=0.95, linewidth=1.5)
            a.text(-0.25 + idx, ctr, ast, rotation=90, ha="center", va="center")

    if errorplot:
        x = [-1]
        y = []
        for idx, decoder, key in zip(range(num_keys), decoders, keys):
            real = decoder.real_performance.get(key)
            null = decoder.null_performance.get(key)
            x.append(idx)
            y.append(np.nanmean(null) * 100 + np.nanstd(null) * 100)
            if idx == 0:
                y.append(np.nanmean(null) * 100 + np.nanstd(null) * 100)
            if idx == num_keys - 1:
                x.append(num_keys)
                y.append(np.nanmean(null) * 100 + np.nanstd(null) * 100)
        for idx, decoder, key in reversed(list(zip(range(num_keys), decoders, keys))):
            real = decoder.real_performance.get(key)
            null = decoder.null_performance.get(key)
            if idx == num_keys - 1:
                x.append(num_keys)
                y.append(np.nanmean(null) * 100 - np.nanstd(null) * 100)
            x.append(idx)
            y.append(np.nanmean(null) * 100 - np.nanstd(null) * 100)
            if idx == 0:
                x.append(-1)
                y.append(np.nanmean(null) * 100 - np.nanstd(null) * 100)

        a.fill(x, y, alpha=0.3, color=(0.65, 0.65, 0.65))

    a.set_frame_on(True)
    plt.setp(a.spines.values(), linewidth=2)
    plt.tight_layout()
