#!/usr/bin/env python3
"""
Simple interface between REINVENT4 and vina_wrapper.py
"""

import sys
import json
import argparse
import signal
import time
from vina_wrapper import VinaWrapper

def timeout_handler(signum, frame):
    raise TimeoutError("Docking timeout")

def dock_with_timeout(smiles, timeout_seconds=30):
    """Dock with timeout to prevent hanging"""
    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        
        config_file = 'data/binding_site.conf'
        pdb_file = 'data/protein_pocket.pdb'
        
        wrapper = VinaWrapper(config_file, pdb_file, smiles)
        docking_score = wrapper.dock()
        
        signal.alarm(0)  # Cancel alarm
        return docking_score
        
    except TimeoutError:
        signal.alarm(0)
        return None
    except Exception:
        signal.alarm(0)
        return None

def main():
    try:
        # Check if we have command line arguments
        parser = argparse.ArgumentParser()
        parser.add_argument('-smiles', help='SMILES string for testing')
        args = parser.parse_args()
        
        # Read SMILES from command line or stdin
        if args.smiles:
            smiles_input = args.smiles
        else:
            smiles_input = sys.stdin.read().strip()
        
        if not smiles_input:
            print('{"version": 1, "payload": {"predictions": []}}')
            sys.stdout.flush()
            return
        
        # Split newline-separated SMILES (REINVENT4 format)
        smiles_list = [s.strip() for s in smiles_input.split('\n') if s.strip()]
        
        scores = []
        
        for smiles in smiles_list:
            try:
                # Quick validity check
                if len(smiles) > 200 or len(smiles) < 3:
                    scores.append(0.0)
                    continue
                
                # Dock with timeout
                docking_score = dock_with_timeout(smiles, timeout_seconds=30)
                
                if docking_score is not None and isinstance(docking_score, (int, float)) and not (docking_score != docking_score):  # Check for NaN
                    # Convert to 0-1 scale (more negative = better = higher score)
                    # Scale: -10 kcal/mol = 1.0 (excellent), -2 kcal/mol = 0.0 (poor)
                    normalized_score = max(0.0, min(1.0, (-docking_score - 2) / 8))
                    # Additional NaN check after normalization
                    if normalized_score != normalized_score:  # NaN check
                        normalized_score = 0.0
                else:
                    normalized_score = 0.0
                    
                scores.append(normalized_score)
                
            except Exception:
                scores.append(0.0)
        
        # Ensure we have same number of scores as input SMILES
        if len(scores) != len(smiles_list):
            scores = [0.0] * len(smiles_list)
        
        # Final NaN check on all scores
        clean_scores = []
        for score in scores:
            if isinstance(score, (int, float)) and not (score != score):  # Valid number, not NaN
                clean_scores.append(float(score))
            else:
                clean_scores.append(0.0)
        
        # Output JSON format for REINVENT4 external_process
        result = {"version": 1, "payload": {"predictions": clean_scores}}
        print(json.dumps(result))
        sys.stdout.flush()
        
    except Exception:
        # Emergency fallback - always return valid JSON
        try:
            smiles_count = len(smiles_input.split(';')) if 'smiles_input' in locals() else 1
            result = {"version": 1, "payload": {"predictions": [0.0] * smiles_count}}
            print(json.dumps(result))
            sys.stdout.flush()
        except:
            print('{"version": 1, "payload": {"predictions": [0.0]}}')
            sys.stdout.flush()

if __name__ == '__main__':
    main()