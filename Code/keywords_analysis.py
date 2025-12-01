#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 30 22:50:30 2025

@author: anetakartali
"""

from adjustText import adjust_text
import ast
from collections import defaultdict
import fnmatch
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
import seaborn as sns

def parse_nodes(x):
    if pd.isna(x) or x == "":
        return []
    # split by semicolon, strip spaces, remove empty pieces
    return [item.strip() for item in str(x).split(";") if item.strip()]

import matplotlib.pyplot as plt
import seaborn as sns

def plot_grouped_keywords(source, grouped_df):
    grouped_df = grouped_df[grouped_df['Group'] != "Ambiguous"]
    df = grouped_df[grouped_df['Source'] == source]
    # Unique domains
    domains = df['Domain'].unique()
    
    # Assign random positions for clusters
    # cluster_positions = {domain: (np.random.rand()*200, np.random.rand()*200) for domain in domains}
    cluster_positions = {
      'Applied computing': (161.13977403164418, 94.76814728505714),
      'Artificial intelligence': (35.01754435398594, 2.484704872687682),
      'Computer Systems Engineering': (15.659081112360095, 36.28116737120379),
      'Computer vision and multimedia computation': (140.23767377073514, 146.8285569779893),
      'Data management and data science': (64.78536306988812, 123.28859051195269),
      'Distributed computing and systems software': (150.0716970935269, 24.28835969406331),
      'Human-centred computing': (6.253792485411247, 153.55847214956736),
      'Machine learning': (72.14563951469594, 65.93426556443171),
      'Software engineering': (85.89740727769544, 17.19667885971039),
      'Theory of computation': (170.677839846511, 180.63052614655683),
      'Cybersecurity and privacy': (85, 170),
      'Graphics, augmented reality and games': (160, 55)
    }
    
    # Normalize bubble sizes
    min_count = df['Count'].min()
    max_count = df['Count'].max()
    size_scale = 2000  # scale factor for plotting
    df['Size'] = df['Count'].apply(lambda x: size_scale * ((x - min_count)/(max_count - min_count) + 0.2))
    
    colors = plt.cm.tab20.colors  # Use a palette of 20 colors
    color_map = {domain: colors[i % len(colors)] for i, domain in enumerate(domains)}

    # Plot
    fig, ax = plt.subplots(figsize=(16, 12))
    for domain in domains:
        cx, cy = cluster_positions[domain]
        plt.text(cx, cy, domain, ha='center', va='center', fontsize=14)
    texts = []
    labels = []
    for _, row in df.iterrows():
        domain = row['Domain']
        cx, cy = cluster_positions[row['Domain']]
        # Randomly jitter positions around the domain center
        # x = cx + np.random.normal(0, 1)
        # y = cy + np.random.normal(0, 1)
        shift_x = 0
        shift_y = 0
        while np.abs(shift_x) < 10 or np.abs(shift_y) < 10:
            shift_x = np.random.uniform(-15, 15)
            shift_y = np.random.uniform(-15, 15) 
        x = cx + shift_x
        y = cy + shift_y
        plt.scatter(x, y, s=row['Size'], color=color_map[domain], alpha=0.6, edgecolors='w', linewidth=0.5, label=domain)
        texts.append(ax.text(x, y, row['Group'], ha='center', va='center', fontsize=10))
        labels.append(domain)
        # plt.text(x, y, row['Group'], ha='center', va='center', fontsize=10)
    
    handles, labels = plt.gca().get_legend_handles_labels()
    # adjust_text(texts, arrowprops=dict(arrowstyle='-', color='white'))
    adjust_text(texts)
    
    # ax.set_title(title, fontsize=20)
    ax.axis('off')
    plt.legend(labels)
    unique_labels = dict(zip(labels, handles))
    lgd = ax.legend(unique_labels.values(), unique_labels.keys(), loc="best")
    for legend_handle in lgd.legend_handles:
        legend_handle.set_sizes([50])
    # plt.show()
    plot_path = f"keyword_maps/{source}_keywords.png"
    plt.savefig(plot_path, dpi=300)


# Load and pre-process the CSV file
results_csv_path = "cfp_paper_matches_CV.csv"
df = pd.read_csv(results_csv_path)

for col in ['CFP_keyword_counts', 'Papers_keyword_counts']:
    df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.strip() else {})
df["Include"] = df["Include"].str.strip().str.lower().map({"yes": True, "no": False})

os.makedirs("keyword_maps", exist_ok=True)

summary_rows = []
keyword_map_rows = []

# Domain analysis -------------------------------------------------------------
for domain, d in df.groupby("Domain"):
    
    d_yes = d[d["Include"]]
    
    # Total number of venues found and the number of venues taken into consideration
    venues_total = len(d)
    venues_included = len(d_yes)
    
    # Number of venues with found keywords in CFPs and in accepted papers
    venues_with_cfp_keywords = sum(d_yes["CFP_keyword_counts"].apply(lambda d: len(d) > 0))
    venues_with_paper_keywords = sum(d_yes["Papers_keyword_counts"].apply(lambda d: len(d) > 0))

    # Percentage of venues with present keywords in CFPs and in accepted papers
    pct_cfp_nonzero = venues_with_cfp_keywords / venues_included * 100 if venues_included else 0
    pct_paper_nonzero = venues_with_paper_keywords / venues_included * 100 if venues_included else 0

    cfp_nodes = set().union(*d_yes["CFP_matched_nodes"].apply(parse_nodes)) if venues_included else set()
    paper_nodes = set().union(*d_yes["Papers_matched_nodes"].apply(parse_nodes)) if venues_included else set()

    # Calculate per-keyword total number
    cfp_keyword_totals = defaultdict(int)
    papers_keyword_totals = defaultdict(int)

    for counts in d_yes["CFP_keyword_counts"]:
        for k, v in counts.items():
            cfp_keyword_totals[k] += v

    for counts in d_yes["Papers_keyword_counts"]:
        for k, v in counts.items():
            papers_keyword_totals[k] += v

    # Save summary row
    summary_rows.append({
        "Domain": domain,
        "Venues in domain": venues_total,
        "Included (Yes)": venues_included,
        "CFP venues with keywords": venues_with_cfp_keywords,
        "Paper venues with keywords": venues_with_paper_keywords,
        "% CFP nonzero": pct_cfp_nonzero,
        "% Paper nonzero": pct_paper_nonzero,
        "CFP unique matched nodes": list(cfp_nodes),
        "Paper unique matched nodes": list(paper_nodes),
    })

    # Keyword dot plot maps ---------------------------------------------------
    for k, v in cfp_keyword_totals.items():
        keyword_map_rows.append({
            "Domain": domain,
            "Source": "CFP",
            "Keyword": k,
            "Count": v
        })

    for k, v in papers_keyword_totals.items():
        keyword_map_rows.append({
            "Domain": domain,
            "Source": "Papers",
            "Keyword": k,
            "Count": v
        })


# Convert to dataframes
summary_df = pd.DataFrame(summary_rows)
keyword_map_df = pd.DataFrame(keyword_map_rows)

# Map identified keywords into predefined groups:
keyword_groups = {
    "Sustainability": [
        "sustain*",
        "eco*",
        "environmental impact",
        "climate",
        "climate-aware"
    ],
    "Carbon emissions": [
        "carbon",
        "CO2",
        "CO₂",
        "carbon-aware",
        "carbon emission",
        "carbon footprint",
        "embodied carbon",
        "emission",
        "net-zero",
        "offset*",
        "carbon accounting",
        "carbon reporting",
        "carbon-report"
    ],
    "Energy efficiency": [
        "energy-eff*",
        "low-power",
        "power-aware",
        "energy-report"
    ],
    "Green": [
        "green",
    ],
    "LCA": [
        "LCA",
        "life cycle",
        "life-cycle"
    ],
    "Renewable Energy": [
        "renewable"
    ],
    "Ambiguous": [
        "energy",
        "environment*",
    ],
    "Performance": [
        "efficien*"
    ]
}

def find_group(keyword, group_dict):
    keyword_lower = keyword.lower()

    for group, patterns in group_dict.items():
        for pattern in patterns:
            if fnmatch.fnmatch(keyword_lower, pattern.lower()):
                return group
    return "Other"  # fallback group

# Assign groups to each keyword
keyword_map_df["Group"] = keyword_map_df["Keyword"].apply(lambda k: find_group(k, keyword_groups))

# Merge rows so counts are grouped
grouped_keyword_map_df = (
    keyword_map_df
    .groupby(["Domain", "Source", "Group"], as_index=False)["Count"]
    .sum()
)

# Create plots
plot_grouped_keywords("CFP", grouped_keyword_map_df)
plot_grouped_keywords("Papers", grouped_keyword_map_df)

# Export to Excel
with pd.ExcelWriter("domain_analysis.xlsx") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    keyword_map_df.to_excel(writer, sheet_name="Keyword_Map", index=False)
    grouped_keyword_map_df.to_excel(writer, sheet_name="Grouped_Keyword_Map", index=False)
    
print("Analysis complete!")
print("Excel file saved as: domain_analysis.xlsx")
print("Plots saved in folder: keyword_maps/")
