# GENIE_TMB

This repository uses TCGA and PCAWG MAF files to calibrate GENIE panel data with the ATGC model: https://github.com/OmnesRes/ATGC.

ATGC can featurize variants using genomic position and sequence context, however it is currently unclear whether these features transfer to non-TCGA or non-PCAWG data, so the model run here conservatively used mutation counts (but can be easily altered to featurize the variants).

The TCGA data is exome data while the PCAWG data is whole genome.  Ideally a model would be trained on a MAF that contained CDS and noncoding mutations (whole genome data), and the panel data would also report all CDS and noncoding mutations.  However, most panels do not report noncoding mutations and the PCAWG data contains limited samples, so we limited ourselves to CDS mutations and merged the TCGA and PCAWG MAFs (processed with process_pcawg_maf.py and process_tcga_mc3_public.py).

Using this merged MAF we then trained models using GENIE panel coordinates.  Unfortunately many GENIE panels only report nonsyn mutations, so we trained models that used all CDS mutations (cds_runs.py) and only nonsyn mutations (nonsyn_runs.py).  Then these models were used to predict the TMB of GENIE samples ( process_genie_maf.py), with the model chosen depending on whether the panel reported all CDS mutations.
