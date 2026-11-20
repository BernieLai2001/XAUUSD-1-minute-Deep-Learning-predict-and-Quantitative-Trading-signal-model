# 报告图的统一样式：颜色、字体、matplotlib rcParams
BLUE, ORANGE, INK, INK2 = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e"

# 中文图用冬青黑体（没有时用 Arial Unicode MS），英文图用 Arial
FONT = {"zh": ["Hiragino Sans GB", "Arial Unicode MS"], "en": ["Arial", "Helvetica"]}

# 浅色背景、浅灰网格、去掉上边框和右边框
CORE_STYLE = {
    "axes.unicode_minus": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "axes.grid": True, "grid.color": "#e8e7e2", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
}

# 在 CORE_STYLE 基础上统一字号、网格放在图形下面、图例无边框
BASE_STYLE = CORE_STYLE | {
    "axes.axisbelow": True,
    "axes.titlesize": 11, "axes.labelsize": 9.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
    "legend.fontsize": 8.5, "legend.frameon": False,
}
