import numpy as np
import pandas as pd

# Theoretical well-mixed sampled abundances (x = 0 for all clones)
def sample_well_mixed(df, C1='count_B', C2='count_T'):
    
    counts_1 = df[C1].to_numpy(dtype=np.int64)
    counts_2 = df[C2].to_numpy(dtype=np.int64)  
    counts_total = counts_1 + counts_2
    
    counts_1_sample  = np.random.binomial(counts_total, 0.5)
    counts_2_sample = counts_total -  counts_1_sample
    
    return pd.DataFrame({C1 :counts_1_sample, C2:counts_2_sample, "count_total": counts_total})
    
# Observed log-ratios
def get_empirical_log_ratios(df, C1='count_B', C2='count_T', threshold=10, base=10):
    
    counts_1 = df[C1].to_numpy(dtype=np.int64)
    counts_2 = df[C2].to_numpy(dtype=np.int64) 
    
    to_keep = ((counts_1 + counts_2) > threshold) & (counts_1 != 0) & (counts_2 != 0)
    
    counts_1 = counts_1[to_keep]
    counts_2 = counts_2[to_keep]
    
    log_ratios = np.log(counts_1/counts_2) / np.log(base)
    
    return log_ratios