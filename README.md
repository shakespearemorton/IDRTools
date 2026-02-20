# IDRTools

A collection of published tools adapted for batch processing of intrinsically disordered protein sequences.

## GIN Cluster Calculator - gin_group.py

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

---

## Phase Separation Predictor - ps_pred.py

Predicts IDR transfer free energies (ΔG) and saturation concentrations using machine learning models trained on CALVADOS 2 simulations at T=293 K, I=150 mM.

**Citation:** von Bülow, S., Tesei, G., Zaidi, F.K., Mittag, T., & Lindorff-Larsen, K. (2025). Prediction of phase-separation propensities of disordered proteins from sequence. *Proc. Natl. Acad. Sci. U.S.A.*

**Usage:**
```bash
# Single sequence
python ps_pred.py -single MADEEKLPPGWEKRMSRSSGRVYYFNHITNASQWERPSGNQ -output results.csv

# Batch processing from FASTA
python ps_pred.py -fasta sequences.fasta -output results.csv

# Batch processing from CSV
python ps_pred.py -sheet sequences.csv -output results.csv -n_cpus 8
```

**Input formats for CSV batch mode:**
- **Option 1:** Columns `seq_name` and `sequence` (or `fasta`)
- **Option 2:** Columns `Uniprot`, `Start Pos`, `End Pos` and `sequence` (or `fasta`)
  - seq_name will be auto-generated as `{Uniprot}_{Start Pos}_{End Pos}`

**Requirements:** `numpy pandas scipy joblib biopython tqdm scikit-learn`

**Note:** First run automatically downloads required model files (~5 MB).

---

## Output Variables

### GIN Cluster Calculator (gin_group.py)

**Core outputs:**
- `seq_name`: Sequence identifier
- `GIN_cluster`: Assigned cluster (0-29), each representing distinct IDR grammars
- `min_inter_cluster_dist`: Distance to nearest other cluster (higher = more confident assignment)

**Z-score features (90 total):**
Each feature quantifies how the IDR sequence deviates from the human IDRome reference:
- **Z-score interpretation:**
  - **|Z| < 1**: Feature is random/typical
  - **|Z| ≥ 1**: Feature is non-random
  - **Z > 0**: Feature is enriched/blocky relative to reference
  - **Z < 0**: Feature is depleted/well-mixed relative to reference
  - **Dynamic range**: Typically -3 to +3 (values are bounded in analysis)

**Compositional features (54):** Amino acid fractions, grouped fractions (charged, polar, hydrophobic, aromatic), patches of specific residues, physicochemical properties (FCR, NCPR, hydropathy, disorder propensity, PPII propensity, isoelectric point)

**Patterning features (36):** Binary mixing parameters (δ) between residue types:
- `pol-pol`, `pol-hyd`, `pol-pos`, `pol-neg`, `pol-aro`, `pol-ala`, `pol-pro`, `pol-gly`
- `hyd-hyd`, `hyd-pos`, `hyd-neg`, `hyd-aro`, `hyd-ala`, `hyd-pro`, `hyd-gly`
- `pos-pos`, `pos-neg`, `pos-aro`, `pos-ala`, `pos-pro`, `pos-gly`
- `neg-neg`, `neg-aro`, `neg-ala`, `neg-pro`, `neg-gly`
- `aro-aro`, `aro-ala`, `aro-pro`, `aro-gly`
- `ala-ala`, `ala-pro`, `ala-gly`
- `pro-pro`, `pro-gly`
- `gly-gly`

Where: pol=polar (S,T,N,Q,C,H), hyd=hydrophobic (I,L,M,V), pos=positive (R,K), neg=negative (E,D), aro=aromatic (F,W,Y), ala=alanine, pro=proline, gly=glycine

### Phase Separation Predictor (ps_pred.py)

**Predictions:**
- `dG_kT`: Transfer free energy from dilute to dense phase (kT units)
  - More negative = stronger phase separation propensity
  - Uncertainty: ±1.0 kT
- `dG_kT_lower`, `dG_kT_upper`: Uncertainty bounds
- `cdil_mgml`: Saturation concentration (mg/mL) with bounds
- `cdil_uM`: Saturation concentration (μM) with bounds

**Molecular properties:**
- `mw_kDa`: Molecular weight (kDa)

**Sequence features used for prediction:**
- `mean_lambda`: Mean hydrophobicity (λ̄) - average residue stickiness
- `faro`: Fraction of aromatic residues (F, W, Y)
- `shd`: Sequence hydropathy decoration - patterning of sticky residues
- `ncpr`: Net charge per residue - (|K+R| - |D+E|) / length
- `fcr`: Fraction of charged residues (K, R, D, E)
- `scd`: Sequence charge decoration - patterning of charged residues
- `ah_ij`: Attractive hydrophobic interactions (config-averaged, volume-scaled)
- `nu_svr`: Predicted Flory scaling exponent (chain compaction measure)