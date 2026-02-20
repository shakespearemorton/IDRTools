#!/usr/bin/env python3
"""
ps_pred - Phase Separation Prediction for Intrinsically Disordered Proteins

This tool predicts IDR transfer free energies (ΔG) and saturation concentrations 
from protein sequences using machine learning models trained on CALVADOS 2 simulations.

Conditions: T=293 K, I=150 mM (fixed)

Reference: S. von Bülow, G. Tesei, F. K. Zaidi, T. Mittag, K. Lindorff-Larsen,
"Prediction of phase-separation propensities of disordered proteins from sequence"
Proc. Natl. Acad. Sci. U.S.A. (2025)
"""

import sys
import os
import argparse
import warnings
import urllib.request
import numpy as np
import pandas as pd
import joblib
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

warnings.simplefilter("ignore")

# GitHub repository URL
GITHUB_BASE = 'https://raw.githubusercontent.com/KULL-Centre/_2024_buelow_PSpred/main'


def download_required_files():
    """Download necessary model files and data from GitHub if missing"""
    files_to_download = {
        'residues.csv': f'{GITHUB_BASE}/data/residues.csv',
        'model_dG.joblib': f'{GITHUB_BASE}/models/idrome90/mlp/dG/model.joblib',
        'model_logcdil_mgml.joblib': f'{GITHUB_BASE}/models/idrome90/mlp/logcdil_mgml/model.joblib',
        'svr_model_nu.joblib': f'{GITHUB_BASE}/models/svr_model_nu.joblib',
        'sequence.py': f'{GITHUB_BASE}/scripts_colab/sequence.py',
        'predictor.py': f'{GITHUB_BASE}/scripts_colab/predictor.py',
    }
    
    # Check which files are missing
    missing_files = [f for f in files_to_download.keys() if not os.path.exists(f)]
    
    if not missing_files:
        return  # All files present, nothing to download
    
    print(f"Downloading {len(missing_files)} required file(s)...")
    for filename in missing_files:
        url = files_to_download[filename]
        try:
            print(f"  - {filename}")
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            print(f"Error downloading {filename}: {e}")
            sys.exit(1)
    print("Files downloaded successfully.")


download_required_files()
import predictor
import sequence
from predictor import X_from_seq


def load_models_and_data():
    """Load ML models and residue data"""
    sys.modules['__main__'].Model = predictor.Model
    sys.modules['__main__'].AttrSetter = predictor.AttrSetter
    residues = pd.read_csv('residues.csv').set_index('one')
    models = {
        'dG': joblib.load('model_dG.joblib'),
        'logcdil_mgml': joblib.load('model_logcdil_mgml.joblib')
    }
    nu_file = 'svr_model_nu.joblib'
    return models, residues, nu_file


def calculate_sequence_features(seq, residues, nu_file, charge_termini=True):
    """
    Calculate all required features for a sequence
    
    Features: mean_lambda, faro, shd, ncpr, fcr, scd, ah_ij, nu_svr
    """
    seqfeats = sequence.SeqFeatures(seq, residues=residues, 
                                   charge_termini=charge_termini, 
                                   nu_file=nu_file)
    
    features = {
        'mean_lambda': seqfeats.mean_lambda,
        'faro': seqfeats.faro,
        'shd': seqfeats.shd,
        'ncpr': seqfeats.ncpr,
        'fcr': seqfeats.fcr,
        'scd': seqfeats.scd,
        'ah_ij': seqfeats.ah_ij,
        'nu_svr': seqfeats.nu_svr,
        'mw': seqfeats.mw  # molecular weight for uM conversion
    }
    
    return features


def predict_single_sequence(seq, models, residues, nu_file, charge_termini=True):
    """
    Predict ΔG and saturation concentration for a single sequence
    
    Returns:
        dict with predictions and molecular weight
    """
    # Calculate features
    seq_features = calculate_sequence_features(seq, residues, nu_file, charge_termini)
    
    # Create feature matrix
    features = ['mean_lambda', 'faro', 'shd', 'ncpr', 'fcr', 'scd', 'ah_ij', 'nu_svr']
    X = X_from_seq(seq, features, residues=residues, 
                   charge_termini=charge_termini, nu_file=nu_file)
    
    results = {'mw': seq_features['mw']}
    
    # Predict ΔG
    dG_predictions = models['dG'].predict(X)
    dG_mean = np.mean(dG_predictions)
    results['dG'] = dG_mean
    results['dG_lower'] = dG_mean - 1.0
    results['dG_upper'] = dG_mean + 1.0
    
    # Predict log(cdil) and convert to mg/mL
    logcdil_predictions = models['logcdil_mgml'].predict(X)
    logcdil_mean = np.mean(logcdil_predictions)
    cdil_mgml = np.exp(logcdil_mean)
    
    results['cdil_mgml'] = cdil_mgml
    results['cdil_mgml_lower'] = np.exp(logcdil_mean - 0.82)
    results['cdil_mgml_upper'] = np.exp(logcdil_mean + 0.82)
    
    # Convert to uM
    results['cdil_uM'] = cdil_mgml / seq_features['mw'] * 1e6
    results['cdil_uM_lower'] = results['cdil_mgml_lower'] / seq_features['mw'] * 1e6
    results['cdil_uM_upper'] = results['cdil_mgml_upper'] / seq_features['mw'] * 1e6
    
    return results


def process_sequence_batch(args):
    """Process a single sequence for batch mode (for multiprocessing)"""
    seq_name, seq, models, residues, nu_file, charge_termini = args
    
    try:
        results = predict_single_sequence(seq, models, residues, nu_file, charge_termini)
        results['seq_name'] = seq_name
        results['sequence'] = seq
        return results
    except Exception as e:
        print(f"Error processing {seq_name}: {str(e)}")
        return None


def read_fasta(fasta_file):
    """Read sequences from FASTA file"""
    from Bio import SeqIO
    
    records = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        records[record.id] = str(record.seq).upper()
    return records


def main():
    parser = argparse.ArgumentParser(
        description='PSLab - Phase Separation Prediction for IDRs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single sequence:
    python pslab_simplified.py -single MADEEKLPPGWEKRMSRSSGRVYYFNHITNASQWERPSGNQ -output results.csv
  
  Batch processing from FASTA:
    python pslab_simplified.py -fasta sequences.fasta -output results.csv
  
  Batch processing from CSV:
    python pslab_simplified.py -sheet sequences.csv -output results.csv
    (input.csv must have 'sequence' or 'fasta' column)
    (input.csv must have 'seq_name' column OR 'Uniprot', 'Start Pos', 'End Pos' columns)

Options:
  -charge_termini: Include N/C terminal charges (default: True)
  -n_cpus: Number of CPUs for parallel processing
        """
    )
    
    parser.add_argument('-single', type=str, metavar='SEQUENCE',
                        help='Single sequence to analyze')
    parser.add_argument('-fasta', type=str, metavar='FASTA_FILE',
                        help='FASTA file with sequences')
    parser.add_argument('-sheet', type=str, metavar='CSV_FILE',
                        help='CSV file with sequence/fasta column and seq_name or (Uniprot, Start Pos, End Pos) columns')
    parser.add_argument('-output', type=str, required=True, metavar='OUTPUT_CSV',
                        help='Output CSV file path')
    parser.add_argument('-charge_termini', type=bool, default=True, metavar='BOOL',
                        help='Include terminal charges (default: True)')
    parser.add_argument('-n_cpus', type=int, default=cpu_count()-1, metavar='N',
                        help=f'Number of CPUs for parallel processing (default: {cpu_count()-1})')
    
    args = parser.parse_args()
    
    # Validate arguments
    mode_count = sum([args.single is not None, args.fasta is not None, args.sheet is not None])
    if mode_count == 0:
        parser.error('One of -single, -fasta, or -sheet must be provided')
    if mode_count > 1:
        parser.error('Only one of -single, -fasta, or -sheet can be used')
    
    # Load models
    models, residues, nu_file = load_models_and_data()
    
    # Process sequences
    if args.single:
        print("Processing single sequence...")
        seq = args.single.upper().replace(' ', '')
        
        results = predict_single_sequence(seq, models, residues, nu_file, args.charge_termini)
        
        print("\n" + "="*80)
        print(f"Sequence: {seq[:50]}{'...' if len(seq) > 50 else ''}")
        print("-"*80)
        print(f"Delta G                   = {results['dG']:5.1f} kT     ({results['dG_lower']:.1f} -- {results['dG_upper']:.1f} kT)")
        print(f"Saturation concentration  = {results['cdil_mgml']:5.1f} mg/mL  ({results['cdil_mgml_lower']:.1f} -- {results['cdil_mgml_upper']:.1f} mg/mL)")
        print(f"                          = {results['cdil_uM']:5.1f} uM     ({results['cdil_uM_lower']:.1f} -- {results['cdil_uM_upper']:.1f} uM)")
        print("="*80 + "\n")
        
        # Save to CSV
        df = pd.DataFrame([{
            'seq_name': 'input_sequence',
            'sequence': seq,
            'dG_kT': results['dG'],
            'cdil_mgml': results['cdil_mgml'],
            'cdil_uM': results['cdil_uM']
        }])
        df.to_csv(args.output, index=False)
        print(f"Results saved to {args.output}")
    
    else:  # Batch mode
        # Load sequences
        if args.fasta:
            print(f"Loading sequences from {args.fasta}...")
            sequences = read_fasta(args.fasta)
        else:  # args.sheet
            print(f"Loading sequences from {args.sheet}...")
            input_df = pd.read_csv(args.sheet)
            
            # Handle flexible sequence column names
            seq_col = None
            if 'sequence' in input_df.columns:
                seq_col = 'sequence'
            elif 'fasta' in input_df.columns:
                seq_col = 'fasta'
            else:
                print("Error: Input CSV must have 'sequence' or 'fasta' column")
                sys.exit(1)
            
            # Handle flexible seq_name or generate from Uniprot + positions
            if 'seq_name' in input_df.columns:
                seq_names = input_df['seq_name'].tolist()
            elif all(col in input_df.columns for col in ['Uniprot', 'Start Pos', 'End Pos']):
                seq_names = [f"{row['Uniprot']}_{row['Start Pos']}_{row['End Pos']}" 
                            for _, row in input_df.iterrows()]
                print("Generated seq_name from Uniprot_StartPos_EndPos")
            else:
                print("Error: Input CSV must have either 'seq_name' column or 'Uniprot', 'Start Pos', 'End Pos' columns")
                sys.exit(1)
            
            sequences = dict(zip(seq_names, input_df[seq_col].str.upper()))
        
        print(f"Processing {len(sequences)} sequences using {args.n_cpus} CPUs...")
        
        # Prepare arguments for multiprocessing
        process_args = [(name, seq, models, residues, nu_file, args.charge_termini) 
                       for name, seq in sequences.items()]
        
        # Process in parallel
        with Pool(args.n_cpus) as pool:
            results = list(tqdm(pool.imap(process_sequence_batch, process_args), 
                              total=len(process_args), desc="Predicting"))
        
        # Filter out failed results
        results = [r for r in results if r is not None]
        
        if not results:
            print("Error: No sequences were successfully processed")
            sys.exit(1)
        
        # Create output dataframe with all features
        output_data = []
        for r in results:
            # Calculate features for this sequence
            seq_features = calculate_sequence_features(r['sequence'], residues, nu_file, args.charge_termini)
            
            output_data.append({
                'seq_name': r['seq_name'],
                'sequence': r['sequence'],
                'dG_kT': r['dG'],
                'dG_kT_lower': r['dG_lower'],
                'dG_kT_upper': r['dG_upper'],
                'cdil_mgml': r['cdil_mgml'],
                'cdil_mgml_lower': r['cdil_mgml_lower'],
                'cdil_mgml_upper': r['cdil_mgml_upper'],
                'cdil_uM': r['cdil_uM'],
                'cdil_uM_lower': r['cdil_uM_lower'],
                'cdil_uM_upper': r['cdil_uM_upper'],
                'mw_kDa': seq_features['mw'],
                'mean_lambda': seq_features['mean_lambda'],
                'faro': seq_features['faro'],
                'shd': seq_features['shd'],
                'ncpr': seq_features['ncpr'],
                'fcr': seq_features['fcr'],
                'scd': seq_features['scd'],
                'ah_ij': seq_features['ah_ij'],
                'nu_svr': seq_features['nu_svr']
            })
        
        df = pd.DataFrame(output_data)
        df.to_csv(args.output, index=False)
        
        # Print summary table
        print("\n" + "="*100)
        print(f"{'Name':20s} {'Sequence':30s} {'ΔG [kT]':>10s} {'C_sat [mg/mL]':>15s} {'C_sat [uM]':>15s}")
        print("="*100)
        for _, row in df.iterrows():
            seq_display = row['sequence'][:27] + '...' if len(row['sequence']) > 30 else row['sequence']
            print(f"{row['seq_name']:20s} {seq_display:30s} {row['dG_kT']:10.1f} {row['cdil_mgml']:15.1f} {row['cdil_uM']:15.1f}")
        print("="*100)
        
        print(f"\nResults saved to {args.output}")
        print(f"Successfully processed {len(results)}/{len(sequences)} sequences")


if __name__ == "__main__":
    main()