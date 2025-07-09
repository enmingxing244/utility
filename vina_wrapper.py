#!/usr/bin/env python3
"""
Clean Vina wrapper for molecular docking.
Takes a configuration file, PDB file, and SMILES string to perform docking.
"""

import os
import sys
import argparse
import subprocess
import tempfile
from pathlib import Path
import re


class VinaWrapper:
    def __init__(self, config_file, pdb_file, smiles, output_dir=None):
        """
        Initialize Vina wrapper.
        
        Args:
            config_file: Path to Vina configuration file
            pdb_file: Path to PDB protein file
            smiles: SMILES string for ligand
            output_dir: Directory for output files (default: current directory)
        """
        self.config_file = Path(config_file)
        self.pdb_file = Path(pdb_file)
        self.smiles = smiles
        self.output_dir = Path(output_dir) if output_dir else Path.cwd()
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data directory for intermediate files
        self.data_dir = self.output_dir / 'data'
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def _parse_config(self):
        """Parse Vina configuration file."""
        config = {}
        with open(self.config_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    config[key.strip()] = value.strip()
        return config
    
    def _smiles_to_pdbqt(self, smiles, output_file):
        """Convert SMILES to PDBQT format using obabel and MGLTools."""
        try:
            # First convert SMILES to PDB with 3D coordinates
            pdb_file_path = Path(output_file).with_suffix('.pdb')
            pdb_file = str(pdb_file_path.resolve())
            cmd = [
                'obabel', '-ismi', '-opdb', '-O', pdb_file,
                '--gen3d', '--minimize'
            ]
            
            process = subprocess.run(
                cmd,
                input=smiles,
                text=True,
                capture_output=True,
                check=True
            )
            
            # Use MGLTools to prepare ligand PDBQT (run from data directory)
            pdb_filename = pdb_file_path.name
            pdbqt_filename = Path(output_file).name
            cmd = [
                '/fs/ess/PAA0203/xing244/packages/mgltools_x86_64Linux2_1.5.7/bin/pythonsh',
                '/fs/ess/PAA0203/xing244/packages/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py',
                '-l', pdb_filename,
                '-o', pdbqt_filename,
                '-A', 'hydrogens'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, cwd=str(self.data_dir))
            
            # Clean up intermediate PDB file
            os.remove(pdb_file)
            
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error converting SMILES to PDBQT: {e}", file=sys.stderr)
            return False
    
    def _pdb_to_pdbqt(self, pdb_file, output_file):
        """Convert PDB to PDBQT format using MGLTools."""
        try:
            # Copy PDB file to data directory if it's not already there
            pdb_path = Path(pdb_file).resolve()
            data_dir_resolved = self.data_dir.resolve()
            if pdb_path.parent != data_dir_resolved:
                import shutil
                temp_pdb = self.data_dir / pdb_path.name
                if not temp_pdb.exists():  # Only copy if doesn't exist
                    shutil.copy2(pdb_file, temp_pdb)
                pdb_filename = pdb_path.name
            else:
                pdb_filename = pdb_path.name
            
            pdbqt_filename = Path(output_file).name
            cmd = [
                '/fs/ess/PAA0203/xing244/packages/mgltools_x86_64Linux2_1.5.7/bin/pythonsh',
                '/fs/ess/PAA0203/xing244/packages/mgltools_x86_64Linux2_1.5.7/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py',
                '-r', pdb_filename,
                '-o', pdbqt_filename,
                '-A', 'checkhydrogens'
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, cwd=str(self.data_dir))
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"Error converting PDB to PDBQT: {e}", file=sys.stderr)
            return False
    
    def _run_vina(self, receptor_pdbqt, ligand_pdbqt, config, output_file):
        """Run Vina docking."""
        try:
            cmd = [
                'vina',
                '--receptor', str(receptor_pdbqt),
                '--ligand', str(ligand_pdbqt),
                '--out', str(output_file),
                '--center_x', str(config.get('center_x', '0.0')),
                '--center_y', str(config.get('center_y', '0.0')),
                '--center_z', str(config.get('center_z', '0.0')),
                '--size_x', str(config.get('size_x', '20.0')),
                '--size_y', str(config.get('size_y', '20.0')),
                '--size_z', str(config.get('size_z', '20.0')),
                '--exhaustiveness', str(config.get('exhaustiveness', '8')),
                '--num_modes', str(config.get('num_modes', '9'))
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return result.stdout
            
        except subprocess.CalledProcessError as e:
            print(f"Error running Vina: {e}", file=sys.stderr)
            print(f"Stderr: {e.stderr}", file=sys.stderr)
            return None
    
    def _extract_best_score(self, vina_output):
        """Extract the best docking score from Vina output."""
        try:
            if not vina_output:
                return None
                
            lines = vina_output.split('\n')
            
            # Method 1: Look for REMARK VINA RESULT in output file
            for line in lines:
                if 'REMARK VINA RESULT:' in line:
                    parts = line.split()
                    if len(parts) >= 4:
                        score = float(parts[3])
                        if not (score != score):  # NaN check
                            return score
            
            # Method 2: Look for docking results table (mode 1 is best)
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('1 ') or stripped.startswith('1\t'):
                    parts = stripped.split()
                    if len(parts) >= 2:
                        score = float(parts[1])
                        if not (score != score):  # NaN check
                            return score
            
            # Method 3: Look for "Affinity" in output
            for i, line in enumerate(lines):
                if 'Affinity' in line and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    if next_line:
                        parts = next_line.split()
                        if len(parts) >= 2:
                            score = float(parts[1])
                            if not (score != score):  # NaN check
                                return score
            
            return None
            
        except (ValueError, IndexError) as e:
            print(f"Error extracting score: {e}", file=sys.stderr)
            return None
    
    def dock(self):
        """
        Perform molecular docking and return the best score.
        
        Returns:
            float: Best docking score (kcal/mol) or None if failed
        """
        try:
            # Parse configuration
            config = self._parse_config()
            
            # Prepare file paths
            receptor_pdbqt = self.data_dir / 'receptor.pdbqt'
            ligand_pdbqt = self.data_dir / 'ligand.pdbqt'
            output_pdbqt = self.data_dir / 'docked.pdbqt'
            
            # Convert PDB to PDBQT
            print("Converting PDB to PDBQT...", file=sys.stderr)
            if not self._pdb_to_pdbqt(self.pdb_file, receptor_pdbqt):
                return None
            
            # Convert SMILES to PDBQT
            print("Converting SMILES to PDBQT...", file=sys.stderr)
            if not self._smiles_to_pdbqt(self.smiles, str(ligand_pdbqt)):
                return None
            
            # Run Vina docking
            print("Running Vina docking...", file=sys.stderr)
            vina_output = self._run_vina(receptor_pdbqt, ligand_pdbqt, config, output_pdbqt)
            
            if vina_output is None:
                return None
            
            # Extract best score
            best_score = self._extract_best_score(vina_output)
            
            if best_score is not None:
                print(f"Best docking score: {best_score} kcal/mol", file=sys.stderr)
            else:
                print("Could not extract docking score", file=sys.stderr)
                print("Vina output preview:", file=sys.stderr)
                print(vina_output[:500] if vina_output else "No output", file=sys.stderr)
            
            return best_score
            
        except Exception as e:
            print(f"Error during docking: {e}", file=sys.stderr)
            return None
        
        finally:
            # Keep PDBQT files in data directory for future use
            pass


def main():
    parser = argparse.ArgumentParser(description='Vina wrapper for molecular docking')
    parser.add_argument('--config', required=True, help='Vina configuration file')
    parser.add_argument('--pdb', required=True, help='PDB protein file')
    parser.add_argument('--smiles', required=True, help='SMILES string for ligand')
    parser.add_argument('--output', default='.', help='Output directory')
    
    args = parser.parse_args()
    
    # Create wrapper instance
    wrapper = VinaWrapper(args.config, args.pdb, args.smiles, args.output)
    
    # Perform docking
    score = wrapper.dock()
    
    if score is not None:
        print(f"Docking completed successfully. Best score: {score} kcal/mol", file=sys.stderr)
        sys.exit(0)
    else:
        print("Docking failed", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()