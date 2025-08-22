import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os, sys

df = pd.read_pickle('/raid/jcintineo/torcnn/eval/nontor2024_pretor2013/all/torp_nontor2024_pretor2013.pkl')

outdir = '/raid/jcintineo/torcnn/eval/nontor2024_pretor2013/all/'

minPreTornado = np.array(df.minPreTornado)
minPreTornado = minPreTornado[minPreTornado > 0]
minPreTornado[minPreTornado > 60] = 60

hist_vals, bin_edges, patches = plt.hist(minPreTornado, bins=20, color='#b3ffb3', edgecolor='#595959')

for v in [15, 30, 45]:
    plt.axvline(x=v, color="#202b6b", linewidth=3)

plt.title('Minutes Pre-tornado')
plt.xlabel('Minutes')
plt.ylabel('Count')
plt.savefig(f"{outdir}/pretor_dist.png", dpi=300, bbox_inches="tight")
print(f"Saved {outdir}/pretor_dist.png")
