import pandas as pd
import numpy as np
import phate
import scprep


sparse=True
T1 = scprep.io.load_10X('data/scRNAseq/T0_1A', sparse=sparse, gene_labels='both')
T2 = scprep.io.load_10X('data/scRNAseq/T2_3B', sparse=sparse, gene_labels='both')
T3 = scprep.io.load_10X('data/scRNAseq/T4_5C', sparse=sparse, gene_labels='both')
T4 = scprep.io.load_10X('data/scRNAseq/T6_7D', sparse=sparse, gene_labels='both')
T5 = scprep.io.load_10X('data/scRNAseq/T8_9E', sparse=sparse, gene_labels='both')

filtered_batches = []
for batch in [T1, T2, T3, T4, T5]:
    batch = scprep.filter.filter_library_size(batch, percentile=20, keep_cells='above')
    batch = scprep.filter.filter_library_size(batch, percentile=75, keep_cells='below')
    filtered_batches.append(batch)
del T1, T2, T3, T4, T5 # removes objects from memory

EBT_counts, sample_labels = scprep.utils.combine_batches(
    filtered_batches,
    ["Day 00-03", "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"],
    append_to_cell_names=True
)
del filtered_batches # removes objects from memory

EBT_counts, sample_labels = scprep.filter.filter_library_size(EBT_counts, sample_labels, cutoff=2000)
EBT_counts = scprep.filter.filter_rare_genes(EBT_counts, min_cells=10)
EBT_counts = scprep.normalize.library_size_normalize(EBT_counts)
mito_genes = scprep.select.get_gene_set(EBT_counts, starts_with="MT-") # Get all mitochondrial genes. There are 14, FYI.
EBT_counts, sample_labels = scprep.filter.filter_gene_set_expression(
    EBT_counts, sample_labels, genes=mito_genes,
    percentile=90, keep_cells='below')
EBT_counts = scprep.transform.sqrt(EBT_counts)

phate_operator = phate.PHATE(n_jobs=-2)

Y_phate = phate_operator.fit_transform(EBT_counts)

# scprep.plot.scatter2d(Y_phate, c=sample_labels, figsize=(12,8), cmap="Spectral",
#                       ticks=False, label_prefix="PHATE")