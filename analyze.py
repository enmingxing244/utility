#!/usr/bin/env python
"""
Results analysis for REINVENT4 pocket-based drug design
Clean, production-ready analysis with comprehensive visualizations
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
import glob
import json
from datetime import datetime

def analyze_optimization_results():
    """Comprehensive analysis of REINVENT4 optimization results"""
    
    print("🧪 REINVENT4 Pocket-Based Drug Design - Results Analysis")
    print("=" * 60)
    print(f"Analysis started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load all result files
    csv_files = glob.glob('pocket_results_*.csv')
    if not csv_files:
        print("❌ No results files found!")
        return None
    
    # Combine all stages
    all_data = []
    for i, file in enumerate(sorted(csv_files), 1):
        try:
            df = pd.read_csv(file)
            df['stage'] = i
            all_data.append(df)
            print(f"✅ Loaded Stage {i}: {len(df)} molecules from {file}")
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
    
    if not all_data:
        print("❌ No valid data loaded!")
        return None
    
    # Combine all data
    df = pd.concat(all_data, ignore_index=True)
    
    # Filter valid molecules
    valid_df = df[df['SMILES_state'] == 1] if 'SMILES_state' in df.columns else df
    
    print(f"\n📊 DATASET OVERVIEW")
    print(f"Total molecules: {len(df)}")
    print(f"Valid molecules: {len(valid_df)} ({len(valid_df)/len(df)*100:.1f}%)")
    print(f"Unique molecules: {valid_df['SMILES'].nunique()}")
    print(f"Stages completed: {df['stage'].max()}")
    
    # Score analysis
    score_col = 'Score' if 'Score' in valid_df.columns else valid_df.columns[3]
    print(f"\n🏆 SCORE ANALYSIS")
    print(f"Score range: {valid_df[score_col].min():.4f} - {valid_df[score_col].max():.4f}")
    print(f"Mean score: {valid_df[score_col].mean():.4f}")
    print(f"Std score: {valid_df[score_col].std():.4f}")
    
    # Component score analysis
    component_cols = [col for col in valid_df.columns if any(x in col.lower() for x in ['qed', 'vina', 'docking', 'mw', 'logp'])]
    if component_cols:
        print(f"\n📈 COMPONENT SCORES")
        for col in component_cols:
            print(f"{col}: mean={valid_df[col].mean():.3f}, max={valid_df[col].max():.3f}")
    
    # Convergence analysis
    if 'step' in valid_df.columns:
        print(f"\n📊 CONVERGENCE ANALYSIS")
        steps = sorted(valid_df['step'].unique())
        convergence_data = []
        for step in [1, 10, 25, 50, 75, 100] + [steps[-1]]:
            if step in steps:
                step_data = valid_df[valid_df['step'] == step]
                mean_score = step_data[score_col].mean()
                max_score = step_data[score_col].max()
                convergence_data.append((step, mean_score, max_score))
                print(f"Step {step:3d}: mean={mean_score:.3f}, max={max_score:.3f}")
    
    # Top molecules analysis
    print(f"\n🥇 TOP 10 MOLECULES")
    top_molecules = valid_df.nlargest(10, score_col)
    
    top_data = []
    for idx, row in top_molecules.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol:
                props = {
                    'rank': len(top_data) + 1,
                    'smiles': row['SMILES'],
                    'score': float(row[score_col]),
                    'stage': int(row['stage']),
                    'step': int(row.get('step', 0)),
                    'mw': Descriptors.MolWt(mol),
                    'logp': Descriptors.MolLogP(mol),
                    'hbd': Descriptors.NumHDonors(mol),
                    'hba': Descriptors.NumHAcceptors(mol),
                    'tpsa': Descriptors.TPSA(mol),
                    'rotatable_bonds': Descriptors.NumRotatableBonds(mol)
                }
                top_data.append(props)
                
                print(f"#{props['rank']}: Score {props['score']:.4f} | Stage {props['stage']} | Step {props['step']}")
                print(f"    MW: {props['mw']:.0f} | LogP: {props['logp']:.1f} | HBD: {props['hbd']} | HBA: {props['hba']}")
                print(f"    {props['smiles']}")
                print()
        except Exception as e:
            print(f"Error analyzing molecule: {e}")
    
    # Create comprehensive visualizations
    create_analysis_plots(valid_df, score_col, component_cols)
    
    # Generate molecule images
    generate_molecule_images(top_data)
    
    # Stage comparison
    if 'stage' in valid_df.columns and valid_df['stage'].nunique() > 1:
        print(f"\n📊 STAGE COMPARISON")
        for stage in sorted(valid_df['stage'].unique()):
            stage_data = valid_df[valid_df['stage'] == stage]
            print(f"Stage {stage}: {len(stage_data)} molecules, "
                  f"mean score: {stage_data[score_col].mean():.3f}, "
                  f"max score: {stage_data[score_col].max():.3f}")
    
    # Save summary statistics
    save_summary_stats(valid_df, score_col, top_data)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"📁 Generated files:")
    print(f"   • comprehensive_analysis.png - Detailed analysis plots")
    print(f"   • top_molecules.png - Top molecule structures") 
    print(f"   • optimization_summary.json - Summary statistics")
    
    return valid_df

def create_analysis_plots(df, score_col, component_cols):
    """Create comprehensive analysis plots"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('REINVENT4 Pocket-Based Drug Design - Results Analysis', fontsize=16, fontweight='bold')
    
    # 1. Score evolution
    if 'step' in df.columns:
        steps = sorted(df['step'].unique())
        mean_scores = [df[df['step'] == s][score_col].mean() for s in steps]
        max_scores = [df[df['step'] == s][score_col].max() for s in steps]
        
        axes[0, 0].plot(steps, mean_scores, 'b-', linewidth=2, label='Mean')
        axes[0, 0].plot(steps, max_scores, 'r--', linewidth=2, label='Max')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Score Convergence')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Score distribution
    axes[0, 1].hist(df[score_col], bins=40, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Score Distribution')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Molecular weight distribution
    mw_values = []
    for smiles in df['SMILES']:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mw_values.append(Descriptors.MolWt(mol))
        except:
            continue
    
    if mw_values:
        axes[0, 2].hist(mw_values, bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[0, 2].set_xlabel('Molecular Weight (Da)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Molecular Weight Distribution')
        axes[0, 2].axvline(250, color='red', linestyle='--', alpha=0.8, label='Target range')
        axes[0, 2].axvline(450, color='red', linestyle='--', alpha=0.8)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    
    # 4. LogP distribution  
    logp_values = []
    for smiles in df['SMILES']:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                logp_values.append(Descriptors.MolLogP(mol))
        except:
            continue
    
    if logp_values:
        axes[1, 0].hist(logp_values, bins=30, alpha=0.7, color='purple', edgecolor='black')
        axes[1, 0].set_xlabel('LogP')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('LogP Distribution')
        axes[1, 0].axvline(1, color='red', linestyle='--', alpha=0.8, label='Target range')
        axes[1, 0].axvline(4, color='red', linestyle='--', alpha=0.8)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    
    # 5. QED vs MW scatter
    qed_values = []
    mw_values_scatter = []
    scores_scatter = []
    
    for idx, row in df.iterrows():
        try:
            mol = Chem.MolFromSmiles(row['SMILES'])
            if mol:
                qed_values.append(Descriptors.qed(mol))
                mw_values_scatter.append(Descriptors.MolWt(mol))
                scores_scatter.append(row[score_col])
        except:
            continue
    
    if qed_values:
        scatter = axes[1, 1].scatter(qed_values, mw_values_scatter, c=scores_scatter, 
                                   cmap='viridis', alpha=0.6, s=20)
        axes[1, 1].set_xlabel('QED Score')
        axes[1, 1].set_ylabel('Molecular Weight')
        axes[1, 1].set_title('QED vs Molecular Weight')
        plt.colorbar(scatter, ax=axes[1, 1], label='Total Score')
    
    # 6. Stage comparison (if multiple stages)
    if 'stage' in df.columns and df['stage'].nunique() > 1:
        stage_scores = [df[df['stage'] == stage][score_col] for stage in sorted(df['stage'].unique())]
        bp = axes[1, 2].boxplot(stage_scores, labels=[f'Stage {i+1}' for i in range(len(stage_scores))])
        axes[1, 2].set_ylabel('Score')
        axes[1, 2].set_title('Score by Stage')
        axes[1, 2].grid(True, alpha=0.3)
    
    # 7. Score vs properties
    if mw_values and logp_values and len(mw_values) == len(logp_values):
        scatter2 = axes[2, 0].scatter(logp_values, mw_values, c=scores_scatter[:len(logp_values)], 
                                    cmap='plasma', alpha=0.6, s=20)
        axes[2, 0].set_xlabel('LogP')
        axes[2, 0].set_ylabel('Molecular Weight')
        axes[2, 0].set_title('LogP vs MW')
        plt.colorbar(scatter2, ax=axes[2, 0], label='Score')
    
    # 8. Validity over time
    if 'step' in df.columns and 'SMILES_state' in df.columns:
        steps = sorted(df['step'].unique())
        validity_rates = []
        for step in steps:
            step_data = df[df['step'] == step]
            validity_rates.append(step_data['SMILES_state'].mean() * 100)
        
        axes[2, 1].plot(steps, validity_rates, 'g-', linewidth=2)
        axes[2, 1].set_xlabel('Step')
        axes[2, 1].set_ylabel('Valid Molecules (%)')
        axes[2, 1].set_title('Molecular Validity')
        axes[2, 1].set_ylim(0, 105)
        axes[2, 1].grid(True, alpha=0.3)
    
    # 9. Component score correlation
    if len(component_cols) >= 2:
        corr_data = df[component_cols + [score_col]].corr()
        im = axes[2, 2].imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        axes[2, 2].set_xticks(range(len(corr_data.columns)))
        axes[2, 2].set_yticks(range(len(corr_data.columns)))
        axes[2, 2].set_xticklabels(corr_data.columns, rotation=45, ha='right')
        axes[2, 2].set_yticklabels(corr_data.columns)
        axes[2, 2].set_title('Score Correlation Matrix')
        
        # Add correlation values
        for i in range(len(corr_data.columns)):
            for j in range(len(corr_data.columns)):
                axes[2, 2].text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                               ha='center', va='center', 
                               color='white' if abs(corr_data.iloc[i, j]) > 0.5 else 'black')
    
    plt.tight_layout()
    plt.savefig('comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

def generate_molecule_images(top_data):
    """Generate images of top molecules"""
    
    if not top_data:
        return
    
    mols = []
    legends = []
    
    for mol_data in top_data[:9]:  # Top 9 for 3x3 grid
        try:
            mol = Chem.MolFromSmiles(mol_data['smiles'])
            if mol:
                mols.append(mol)
                legend = (f"#{mol_data['rank']}: {mol_data['score']:.4f}\n"
                         f"MW: {mol_data['mw']:.0f} | LogP: {mol_data['logp']:.1f}\n"
                         f"Stage {mol_data['stage']}")
                legends.append(legend)
        except Exception as e:
            print(f"Error generating image for molecule {mol_data['rank']}: {e}")
    
    if mols:
        img = Draw.MolsToGridImage(mols, molsPerRow=3, subImgSize=(400, 400), 
                                  legends=legends, legendFontSize=12)
        img.save('top_molecules.png')
        print(f"Generated images for {len(mols)} top molecules")

def save_summary_stats(df, score_col, top_data):
    """Save summary statistics to JSON"""
    
    summary = {
        'analysis_date': datetime.now().isoformat(),
        'dataset_info': {
            'total_molecules': len(df),
            'unique_molecules': df['SMILES'].nunique(),
            'stages': int(df['stage'].max()) if 'stage' in df.columns else 1,
            'validity_rate': float(df['SMILES_state'].mean()) if 'SMILES_state' in df.columns else 1.0
        },
        'score_statistics': {
            'mean': float(df[score_col].mean()),
            'std': float(df[score_col].std()),
            'min': float(df[score_col].min()),
            'max': float(df[score_col].max()),
            'median': float(df[score_col].median())
        },
        'top_molecules': top_data[:10]  # Top 10
    }
    
    with open('optimization_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    analyze_optimization_results()