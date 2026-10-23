"""Shared figure style (validated categorical palette, recessive axes)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300']
SEQ = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95',
       '#0d366b']
INK, INK2, GRID = '#1f1f1e', '#5f5e58', '#e4e3dd'


def setup():
    plt.rcParams.update({
        'font.size': 9, 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
        'xtick.color': INK2, 'ytick.color': INK2, 'axes.grid': True,
        'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
        'axes.spines.right': False, 'lines.linewidth': 2,
        'legend.frameon': False, 'savefig.dpi': 200,
        'savefig.bbox': 'tight', 'figure.facecolor': 'white'})


def wilson(k, n, z=1.96):
    import numpy as np
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.where(n > 0, k / np.maximum(n, 1), np.nan)
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, c - h, c + h
