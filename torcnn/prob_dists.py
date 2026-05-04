import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import wasserstein_distance

def draw_ridgeline(data, column, group_col, order, title, palette, filename):
    # Filter out missing values (like -99900)
    plot_df = data[data[group_col].isin(order)].copy()
    
    # Set up the FacetGrid: one row per group
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(plot_df, row=group_col, hue=group_col, palette=palette,
                      aspect=5, height=1, row_order=order)

    # Draw the densities (the "Half-Violin" look)
    g.map(sns.kdeplot, column, fill=True, alpha=0.7, lw=1.5, clip=(0,1))
    g.map(sns.kdeplot, column, color="black", lw=1, clip=(0,1)) # Outline

    # Add a median line for each group
    def draw_stats(x, **kwargs):
        # Calculate stats
        q25 = x.quantile(0.25)
        q50 = x.median()
        q75 = x.quantile(0.75)
        # 25% and 75% lines (thinner, lighter)
        plt.axvline(q25, color='black', linestyle=':', lw=1, alpha=0.6)
        plt.axvline(q75, color='black', linestyle=':', lw=1, alpha=0.6)
        # Median line (bold)
        plt.axvline(q50, color='black', linestyle='--', lw=2)
    g.map(draw_stats, column)

    # Clean up the styling to make it look like a vertical list
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    # Add labels to each row manually
    for i, ax in enumerate(g.axes.flat):
        ax.text(-0.05, 0.2, order[i], fontweight="bold", color="black",
                ha="right", va="center", transform=ax.transAxes)

    plt.subplots_adjust(hspace=-0.3) # Overlap the rows slightly for "Raincloud" look
    g.fig.suptitle(title, fontsize=16)
    g.set_xlabels("Predicted Probability")
    
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    print(f"Saved {filename}")


# Load data
outdir = indir = '/work2/jcintineo/torcnn/tests/2011-19/test04/eval2018'
df = pd.read_csv(f'{indir}/eval_data.csv')

os.makedirs(outdir, exist_ok=True)

# --- DATA CLEANING ---
# Handle the '[1]' or '[0]' format if they came in as strings/lists
if df['tornado'].dtype == object:
    df['tornado'] = df['tornado'].str.extract('(\d+)').astype(int)

# --- PLOT 1: TORNADO VS NON-TORNADO DISTRIBUTIONS ---
plt.figure(figsize=(10, 6))
sns.histplot(data=df[df['tornado'] == 0], x='prob', color='blue', label='Non-Tornado', 
             stat='density', kde=False, alpha=0.4, element="step")
sns.histplot(data=df[((df['tornado'] == 1) & (df['pretorMinutes'] < 1))], x='prob', color='red', label='Tornado', 
             stat='density', kde=False, alpha=0.4, element="step")

plt.title('Normalized Probability Distribution: Tornado vs. Non-Tornado')
plt.xlabel('Model Probability')
plt.ylabel('Density')
plt.legend(fontsize='medium')
plt.savefig(f'{outdir}/dist_tor_vs_nontor.png', dpi=300)
print(f"Saved {outdir}/dist_tor_vs_nontor.png")

# --- PLOT 2: PRETOR MINUTES VIOLIN PLOT ---
# Filter for valid pre-tornado minutes (> 0)
pretor_df = df[df['pretorMinutes'] > 0].copy()

# Define the bins
def bin_pretor(mins):
    if 1 <= mins <= 10: return '1-10 min'
    if 11 <= mins <= 20: return '11-20 min'
    if mins >= 21: return '21+ min'
    return None

pretor_df['pretor_group'] = pretor_df['pretorMinutes'].apply(bin_pretor)
# Ensure correct ordering on the X-axis
pretor_order = ['1-10 min', '11-20 min', '21+ min']

draw_ridgeline(pretor_df,
               'prob',
               'pretor_group',
               pretor_order, 
               'Probability Distribution by Lead Time',
               'plasma',
               f'{outdir}/violin_pretorMinutes.png'
)


# --- PLOT 3: MAGTORNADO VIOLIN PLOT ---
# Filter for valid magnitude (>= 0)
mag_df = df[df['magtornado'] >= 0].copy()

# Group >= 4 into a single category
mag_df['mag_group'] = mag_df['magtornado'].apply(lambda x: 'EF4+' if x >= 4 else f'EF{x}')
mag_order = ['EF0', 'EF1', 'EF2', 'EF3', 'EF4+']

draw_ridgeline(mag_df,
               'prob',
               'mag_group',
               mag_order, 
               'Probability Distribution by Intensity',
               'viridis',
               f'{outdir}/violin_magTornado.png'
)


# Wasserstein distances

tor_probs = df[df['tornado'] == 1]['prob']
nontor_probs = df[df['tornado'] == 0]['prob']
wd1 = wasserstein_distance(tor_probs, nontor_probs)
print(f'Wasserstein distance (tor vs nontor): {wd1}')
tor_probs = df[df['pretorMinutes'] > 0]['prob']
nontor_probs = df[df['tornado'] == 0]['prob']
wd2 = wasserstein_distance(tor_probs, nontor_probs)
print(f'Wasserstein distance (pretor vs nontor): {wd2}')
