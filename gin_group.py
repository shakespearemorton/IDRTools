#!/usr/bin/env python3
"""
NARDINI+ z-score and GIN cluster calculator

This tool computes compositional and sequence patterning features for intrinsically
disordered protein sequences and assigns them to GIN (Gaussian Interaction Network)
clusters based on the human proteome reference.

Reference: [K.M. Ruff, M.R. King, A.W. Ying, V. Liu, A. Pant, W.E. Lieberman, M.K. Shinn, X. Su, C. Kadoch, R.V. Pappu. (2025). 
Molecular grammars of predicted intrinsically disordered regions that span the human proteome. Cell.]
(https://www.cell.com/cell/fulltext/S0092-8674(25)01191-2)
"""
import sys
import argparse
import numpy as np
import pandas as pd
import random
import math
import scipy.stats as stats
from localcider.sequenceParameters import SequenceParameters
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import re


def get_patches(seq, aa, numInt=2, minBlockLen=4):
    """Calculate patch fraction for a given amino acid"""
    justKs = '0' * len(seq)
    pos = [i for i, ltr in enumerate(seq) if ltr == aa]
    pos2 = pos.copy()
    for p in range(len(pos) - 1):
        tdi = pos[p + 1] - pos[p]
        if 1 < tdi <= numInt + 1:
            pos2.extend(range(pos[p] + 1, pos[p + 1]))
    justKs = list(justKs)
    for p in pos2:
        justKs[p] = '1'
    justKs = ''.join(justKs)
    the_ones = re.findall(r"1+", justKs)
    idx_ones = [[m.start(0), m.end(0)] for m in re.finditer(r"1+", justKs)]
    patchescombined = ''
    for count, o in enumerate(the_ones):
        myrange = idx_ones[count]
        subseq = seq[myrange[0]:myrange[1]]
        pos3 = [i for i, ltr in enumerate(subseq) if ltr == aa]
        if len(pos3) >= minBlockLen:
            patchescombined += subseq
    return len(patchescombined) / len(seq) if seq else 0


def get_rg_frac(seq, numInt=2):
    """Calculate RG fraction"""
    justKs = '0' * len(seq)
    pos = [i for i, ltr in enumerate(seq) if ltr in ['R', 'G']]
    pos2 = pos.copy()
    for p in range(len(pos) - 1):
        tdi = pos[p + 1] - pos[p]
        if 1 < tdi <= numInt + 1:
            pos2.extend(range(pos[p] + 1, pos[p + 1]))
    justKs = list(justKs)
    for p in pos2:
        justKs[p] = '1'
    justKs = ''.join(justKs)
    the_ones = re.findall(r"1+", justKs)
    idx_ones = [[m.start(0), m.end(0)] for m in re.finditer(r"1+", justKs)]
    patchescombined = ''
    for count, o in enumerate(the_ones):
        myrange = idx_ones[count]
        subseq = seq[myrange[0]:myrange[1]]
        pos3 = subseq.count('RG')
        if pos3 >= 2:
            patchescombined += subseq
    return len(patchescombined) / len(seq) if seq else 0


def get_kappa(seq, type1, type2):
    """Calculate kappa mixing parameter (for different types)"""
    blobsz = [5]
    kappab = []
    for b in blobsz:
        count1 = sum(seq.count(res) for res in type1)
        count2 = sum(seq.count(res) for res in type2)
        
        sigAll = ((count1/len(seq)) - (count2/len(seq)))**2 / ((count1/len(seq)) + (count2/len(seq)))
        
        sigX = []
        for x in range(0, len(seq)-b+2):
            subseq = seq[x:x+b]
            count1 = sum(subseq.count(res) for res in type1)
            count2 = sum(subseq.count(res) for res in type2)
            
            if count1 + count2 == 0:
                sigX.append(0)
            else:
                sigX.append(((count1/b) - (count2/b))**2 / ((count1/b) + (count2/b)))
        
        asym = [(sigX[x] - sigAll)**2 for x in range(len(sigX))]
        kappab.append(np.mean(asym))
    
    return np.mean(kappab)


def get_omega(seq, type1):
    """Calculate omega parameter (for same type self-interaction)"""
    blobsz = [5]
    omegab = []
    for b in blobsz:
        count = sum(seq.count(res) for res in type1)
        sigAll = ((count/len(seq)) - (1 - (count/len(seq))))**2
        
        sigX = []
        for x in range(0, len(seq)-b+2):
            subseq = seq[x:x+b]
            count = sum(subseq.count(res) for res in type1)
            sigX.append(((count/b) - (1 - (count/b)))**2)
        
        asym = [(sigX[x] - sigAll)**2 for x in range(len(sigX))]
        omegab.append(np.mean(asym))
    
    return np.mean(omegab)


def calc_features(seq, num_scrambles=100000):
    """Calculate all compositional and patterning features"""
    seqOb = SequenceParameters(seq)
    aas = 'ACDEFGHIKLMNPQRSTVY'
    aafrac = seqOb.get_amino_acid_fractions()
    slen = seqOb.get_length()
    
    # Compositional features (order matches original code)
    comp_vals = [
        aafrac['A'], aafrac['C'], aafrac['D'], aafrac['E'], aafrac['F'],
        aafrac['G'], aafrac['H'], aafrac['I'], aafrac['K'], aafrac['L'],
        aafrac['M'], aafrac['N'], aafrac['P'], aafrac['Q'], aafrac['R'],
        aafrac['S'], aafrac['T'], aafrac['V'], aafrac['W'], aafrac['Y'],
        aafrac['K']+aafrac['R'],
        aafrac['D']+aafrac['E'],
        aafrac['Q']+aafrac['N']+aafrac['S']+aafrac['T']+aafrac['G']+aafrac['C']+aafrac['H'],
        aafrac['A']+aafrac['L']+aafrac['M']+aafrac['I']+aafrac['V'],
        aafrac['F']+aafrac['W']+aafrac['Y'],
        np.log10(((slen*aafrac['R'])+1)/((slen*aafrac['K'])+1)),
        np.log10(((slen*aafrac['E'])+1)/((slen*aafrac['D'])+1)),
        seqOb.get_fraction_expanding(),
        seqOb.get_FCR(),
        seqOb.get_NCPR(),
        seqOb.get_mean_hydropathy(),
        seqOb.get_fraction_disorder_promoting(),
        seqOb.get_isoelectric_point(),
        seqOb.get_PPII_propensity(mode='hilser')
    ]
    
    for aa in aas:
        comp_vals.append(get_patches(seq, aa))
    comp_vals.append(get_rg_frac(seq))
    
    # Patterning features using NARDINI method
    pol = ['S','T','N','Q','C','H']
    hyd = ['I','L','M','V']
    pos = ['R','K']
    neg = ['E','D']
    aro = ['F','W','Y']
    ala = ['A']
    pro = ['P']
    gly = ['G']
    typeall = [pol, hyd, pos, neg, aro, ala, pro, gly]
    
    fracsall = [sum(seq.count(res) for res in type1)/len(seq) for type1 in typeall]
    
    scr_seq_arr = np.zeros((len(typeall), len(typeall)))
    for count1, type1 in enumerate(typeall):
        for count2, type2 in enumerate(typeall):
            if count2 >= count1:
                if type1 == type2 and fracsall[count1] > 0.10:
                    scr_seq_arr[count1, count2] = get_omega(seq, type1)
                elif type1 != type2 and fracsall[count1] > 0.10 and fracsall[count2] > 0.10:
                    scr_seq_arr[count1, count2] = get_kappa(seq, type1, type2)
    
    myarr = scr_seq_arr.reshape([1, len(typeall)**2])
    
    # Calculate scrambles
    scr_vals = np.zeros((num_scrambles, len(typeall)**2))
    goodrows = []
    for x in range(num_scrambles):
        currseq = ''.join(random.sample(seq, len(seq)))
        scr_seq_arr = np.zeros((len(typeall), len(typeall)))
        for count1, type1 in enumerate(typeall):
            for count2, type2 in enumerate(typeall):
                if count2 >= count1:
                    if x == 0:
                        rowcount = count1 * len(typeall) + count2
                    if type1 == type2 and fracsall[count1] > 0.10:
                        scr_seq_arr[count1, count2] = get_omega(currseq, type1)
                        if x == 0:
                            goodrows.append(rowcount)
                    elif type1 != type2 and fracsall[count1] > 0.10 and fracsall[count2] > 0.10:
                        scr_seq_arr[count1, count2] = get_kappa(currseq, type1, type2)
                        if x == 0:
                            goodrows.append(rowcount)
        scr_vals[x, :] = scr_seq_arr.reshape([1, len(typeall)**2])
    
    # Fit to gamma distribution
    scr_vals_t = scr_vals.transpose()
    amean = []
    avar = []
    for i in range(scr_vals_t.shape[0]):
        if i in goodrows:
            try:
                if np.std(scr_vals_t[i, :]) < 1e-10:
                    amean.append(np.mean(scr_vals_t[i, :]))
                    avar.append(1e-10)
                else:
                    fit_alpha, fit_loc, fit_beta = stats.gamma.fit(scr_vals_t[i, :])
                    cmean = stats.gamma.mean(fit_alpha, fit_loc, fit_beta)
                    cvar = stats.gamma.var(fit_alpha, fit_loc, fit_beta)
                    amean.append(cmean)
                    avar.append(cvar)
            except:
                amean.append(np.mean(scr_vals_t[i, :]))
                var = np.var(scr_vals_t[i, :])
                avar.append(var if var > 1e-10 else 1e-10)
        else:
            amean.append(np.nan)
            avar.append(np.nan)
    
    # Calculate z-scores for patterning
    patt_zscores = []
    for p1 in range(len(typeall)):
        for p2 in range(p1, len(typeall)):
            currpos = p1*len(typeall)+p2
            if myarr[0, currpos] == 0:
                patt_zscores.append(0)
            else:
                patt_zscores.append((myarr[0, currpos]-amean[currpos])/math.sqrt(avar[currpos]))
    
    return comp_vals, patt_zscores


def get_feature_names():
    """Return feature names in the correct order"""
    # Patterning feature names
    pattnames = ['pol', 'hyd', 'pos', 'neg', 'aro', 'ala', 'pro', 'gly']
    patt_feature_names = []
    for i in range(len(pattnames)):
        for j in range(i, len(pattnames)):
            patt_feature_names.append(f'{pattnames[i]}-{pattnames[j]}')
    
    # Compositional feature names
    url = 'https://docs.google.com/spreadsheets/d/1yxt0R1G0gdI2bGpjYXgk_h7-1qA6EY6J/gviz/tq?tqx=out:csv&sheet=sapiens'
    species_df = pd.read_csv(url)
    comp_feature_names = species_df['Feature'].tolist()
    
    return patt_feature_names + comp_feature_names, species_df


def process_sequence(args):
    """Process a single sequence (for multiprocessing)"""
    seq_name, seq, species_df, cent_df = args
    
    try:
        comp_vals, patt_zscores = calc_features(seq, num_scrambles=100000)
        
        # Compositional z-scores
        comp_means = species_df['Mean'].values
        comp_stds = species_df['Std'].values
        comp_zscores = [(comp_vals[i] - comp_means[i]) / comp_stds[i] for i in range(len(comp_vals))]
        
        # Combine z-scores
        all_zscores = patt_zscores + comp_zscores
        all_zscores_bound = [max(-3, min(3, z)) for z in all_zscores]
        
        # Calculate GIN cluster
        cent_mat = cent_df.iloc[:, 1:].values
        dists = [np.sum((cent_mat[c] - all_zscores_bound) ** 2) for c in range(len(cent_mat))]
        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]
        other_dists = [d - min_dist for i, d in enumerate(dists) if i != min_idx]
        min_inter_dist = min(other_dists)
        
        result = {'seq_name': seq_name, 'GIN_cluster': min_idx, 'min_inter_cluster_dist': min_inter_dist}
        for i, z in enumerate(all_zscores):
            result[f'z_{i}'] = z
        
        return result
    except Exception as e:
        print(f"Error processing {seq_name}: {str(e)}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='NARDINI+ z-score and GIN cluster calculator',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single sequence:
    python gin_group.py -single MADEEKLPPGWEKRMSRSSGRVYYFNHITNASQWERPSGNQ -output results.csv
  
  Batch processing:
    python gin_group.py -sheet input.csv -output results.csv
    (input.csv must have 'seq_name' and 'fasta' columns)
        """
    )
    
    parser.add_argument('-single', type=str, metavar='SEQUENCE',
                        help='Single FASTA sequence to analyze')
    parser.add_argument('-sheet', type=str, metavar='CSV_FILE',
                        help='CSV file with seq_name and fasta columns')
    parser.add_argument('-output', type=str, required=True, metavar='OUTPUT_CSV',
                        help='Output CSV file path')
    parser.add_argument('-n_cpus', type=int, default=cpu_count()-1, metavar='N',
                        help=f'Number of CPUs for parallel processing (default: {cpu_count()-1})')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.single and not args.sheet:
        parser.error('Either -single or -sheet must be provided')
    if args.single and args.sheet:
        parser.error('Cannot use both -single and -sheet')
    
    # Load reference data
    print("Loading reference data...")
    all_feature_names, species_df = get_feature_names()
    cent_url = 'https://docs.google.com/spreadsheets/d/1yxt0R1G0gdI2bGpjYXgk_h7-1qA6EY6J/gviz/tq?tqx=out:csv&sheet=GIN_centroids'
    cent_df = pd.read_csv(cent_url)
    
    # Process sequences
    if args.single:
        print("Processing single sequence...")
        seq = args.single.upper()
        result = process_sequence(('input_sequence', seq, species_df, cent_df))
        
        if result:
            # Rename z-scores with feature names
            result_final = {'seq_name': result['seq_name'], 
                           'GIN_cluster': result['GIN_cluster'],
                           'min_inter_cluster_dist': result['min_inter_cluster_dist']}
            for i, name in enumerate(all_feature_names):
                result_final[name] = result[f'z_{i}']
            
            df = pd.DataFrame([result_final])
            df.to_csv(args.output, index=False)
            print(f"\nResults saved to {args.output}")
            print(f"GIN Cluster: {result['GIN_cluster']}")
            print(f"Min Inter-Cluster Distance: {result['min_inter_cluster_dist']:.4f}")
        else:
            print("Error processing sequence")
            sys.exit(1)
    
    else:  # -sheet
        print(f"Loading sequences from {args.sheet}...")
        input_df = pd.read_csv(args.sheet)
        
        if 'seq_name' not in input_df.columns or 'fasta' not in input_df.columns:
            print("Error: Input CSV must have 'seq_name' and 'fasta' columns")
            sys.exit(1)
        
        # Prepare arguments for multiprocessing
        process_args = [(row['seq_name'], row['fasta'].upper(), species_df, cent_df) 
                       for _, row in input_df.iterrows()]
        
        print(f"Processing {len(process_args)} sequences using {args.n_cpus} CPUs...")
        with Pool(args.n_cpus) as pool:
            results = list(tqdm(pool.imap(process_sequence, process_args), 
                              total=len(process_args), desc="Computing GIN clusters"))
        
        # Filter out failed results
        results = [r for r in results if r is not None]
        
        if not results:
            print("Error: No sequences were successfully processed")
            sys.exit(1)
        
        # Rename z-scores with feature names
        results_final = []
        for result in results:
            result_final = {'seq_name': result['seq_name'], 
                          'GIN_cluster': result['GIN_cluster'],
                          'min_inter_cluster_dist': result['min_inter_cluster_dist']}
            for i, name in enumerate(all_feature_names):
                result_final[name] = result[f'z_{i}']
            results_final.append(result_final)
        
        df = pd.DataFrame(results_final)
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")
        print(f"Successfully processed {len(results)}/{len(process_args)} sequences")


if __name__ == "__main__":
    main()