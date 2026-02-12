import pandas as pd
import numpy as np
import time
from tqdm import tqdm

def calculateDistVec(lat1, lon1, lat2, lon2, labels = [], limit = None):
   R = 6373.0
   rlat1 = np.radians(lat1)
   rlon1 = np.radians(lon1)
   rlat2 = np.radians(lat2)
   rlon2 = np.radians(lon2)
   dlat = rlat2 - rlat1
   dlon = rlon2 - rlon1
   a = np.sin(dlat / 2)**2 + np.cos(rlat1) * np.cos(rlat2) * np.sin(dlon / 2)**2
   c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
   distance = np.round(R * c, 3)
   if len(labels) > 0:
      labels = np.array(labels)
      pairedResult = np.column_stack((distance, labels))
      if limit:
         pairedResult = pairedResult[pairedResult[:, 0] < limit]
      sortedResult = pairedResult[pairedResult[:, 0].argsort()]
      return sortedResult
   return distance



dfo = pd.read_csv('/work2/jcintineo/TORP/2022/202206_tracks_WarningReportInfo.csv')

t0=time.time()
dfnontor = dfo[dfo["tornado"] == 0].copy()
dftor = dfo[dfo["tornado"] == 1].copy()
if len(dfnontor) != 0:
   print("Calculating distance from tornadoes:")

   # Sort dftor by time for fast range filtering
   dftor_sorted = dftor.sort_values("TimeUnix").reset_index(drop=True).copy()
   tor_times = dftor_sorted["TimeUnix"].values

   # Prepare output arrays
   tornadoesWithin50kmAnd1hr = []
   closedist = []
   closediff = []

   for lat, lon, nt in tqdm(zip(dfnontor["Lat"], dfnontor["Lon"], dfnontor["TimeUnix"]), total=len(dfnontor)):
      if pd.isnull(lat) or pd.isnull(lon) or pd.isnull(nt):
         tornadoesWithin50kmAnd1hr.append(np.nan)
         closedist.append(np.nan)
         closediff.append(np.nan)
         continue

      # Time filtering using searchsorted
      idx_start = np.searchsorted(tor_times, nt - 3600, side='left')
      idx_end = np.searchsorted(tor_times, nt + 3600, side='right')
      sub_tor = dftor_sorted.iloc[idx_start:idx_end].copy()

      if sub_tor.empty:
         tornadoesWithin50kmAnd1hr.append(np.nan)
         closedist.append(np.nan)
         closediff.append(np.nan)
         continue

      dists = calculateDistVec(lat, lon, sub_tor["Lat"].values, sub_tor["Lon"].values)
      tdiffs = np.abs(sub_tor["TimeUnix"].values - nt)

      mask = dists <= 50
      if not np.any(mask):
         tornadoesWithin50kmAnd1hr.append(np.nan)
         closedist.append(np.nan)
         closediff.append(np.nan)
         continue

      matches = sub_tor.loc[mask].copy()
      matches["dists"] = dists[mask]
      matches["tdiff"] = tdiffs[mask]

      tornadoesWithin50kmAnd1hr.append(";;".join(set(matches["ID"].astype(str))))
      top_spatial = matches.nsmallest(10, "dists")
      closedist.append(";;".join(f"{d:.3f}:{int(t)//60}" for d, t in zip(top_spatial["dists"], top_spatial["tdiff"])))
      top_temporal = matches.nsmallest(10, "tdiff")
      closediff.append(";;".join(f"{d:.3f}:{int(t)//60}" for d, t in zip(top_temporal["dists"], top_temporal["tdiff"])))

   dfo.loc[dfo["tornado"] == 0, "tornadoesWithin50kmAnd1hr"] = tornadoesWithin50kmAnd1hr
   dfo.loc[dfo["tornado"] == 0, "closestTorPointsInSpace"]    = closedist
   dfo.loc[dfo["tornado"] == 0, "closestTorPointsInTime"]     = closediff
   dfo.loc[dfo["tornado"] == 1, "tornadoesWithin50kmAnd1hr"] = np.nan
   dfo.loc[dfo["tornado"] == 1, "closestTorPointsInSpace"]    = np.nan
   dfo.loc[dfo["tornado"] == 1, "closestTorPointsInTime"]     = np.nan

print(time.time() - t0)
print(dfo["closestTorPointsInSpace"].isna().sum())
print(dfo["closestTorPointsInSpace"].isna().sum()/len(dfo))

