# IDRTools

A collection of published tools adapted for batch processing of intrinsically disordered protein sequences.

## GIN Cluster Calculator

Assigns IDP sequences to GIN (Grammers in NARDINI+) clusters based on compositional and patterning features.

**Citation:** Ruff, K.M., King, M.R., Ying, A.W., Liu, V., Pant, A., Lieberman, W.E., Shinn, M.K., Su, X., Kadoch, C., & Pappu, R.V. (2025). Molecular grammars of predicted intrinsically disordered regions that span the human proteome. *Cell*. [https://www.cell.com/cell/fulltext/S0092-8674(25)01191-2](https://www.cell.com/cell/fulltext/S0092-8674(25)01191-2)

**Usage:**
```bash
# Single sequence
python gin_group.py -single MADEEKLPPGWEKRMSRSSGRVYYFNHITNASQWERPSGNQ -output results.csv

# Batch processing (CSV with seq_name and fasta columns)
python gin_group.py -sheet input.csv -output results.csv
```

**Requirements:** `numpy pandas scipy localcider tqdm`
