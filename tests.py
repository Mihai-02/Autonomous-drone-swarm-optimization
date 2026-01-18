import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime
from boustrophedon import BoustrophedonSimulation
from inverse_aco import InverseACOSimulation
import math

class ExperimentRunner:
    def __init__(self, output_dir='results'):
        self.output_dir = output_dir
        self.results = {}
        
    def run_baseline(self, trials=5):
        """Experiment 1: Baseline Performance"""
        print("\n" + "="*60)
        print("EXPERIMENT 1: Baseline Performance")
        print("="*60)
        
        results = {'boustrophedon': [], 'inverse_aco': []}
        
        for trial in range(trials):
            print(f"\nTrial {trial+1}/{trials}")
            
            # Boustrophedon
            print("  Running Boustrophedon...")
            sim_b = BoustrophedonSimulation(grid_size=100, num_drones=25, cell_divisions=5)
            sim_b.start()
            
            results['boustrophedon'].append({
                'steps': sim_b.step_count,
                'coverage': sim_b.coverage,
                'redundancy': sim_b.redundancy
            })
            print(f"    Steps: {sim_b.step_count}, Coverage: {sim_b.coverage}%")
            
            # Inverse ACO
            print("  Running Inverse ACO...")
            sim_i = InverseACOSimulation(grid_size=100, num_drones=25, evaporation_rate=0.005)
            sim_i.start()
            
            results['inverse_aco'].append({
                'steps': sim_i.step_count,
                'coverage': sim_i.coverage,
                'redundancy': sim_i.redundancy
            })
            print(f"    Steps: {sim_i.step_count}, Coverage: {sim_i.coverage}%")
        
        self.results['baseline'] = results
        self._save_results('baseline')
        self._plot_baseline(results)
    
    def run_scalability_drones(self):
        """Experiment 3: Scalability (Drone Count)"""
        print("\n" + "="*60)
        print("EXPERIMENT 3: Scalability (Drone Count)")
        print("="*60)
        
        drone_counts = [16, 25, 36, 49]
        results = {'boustrophedon': {}, 'inverse_aco': {}}
        
        for num_drones in drone_counts:
            print(f"\nTesting with {num_drones} drones...")
            
            # Boustrophedon
            sim_b = BoustrophedonSimulation(grid_size=100, num_drones=num_drones, cell_divisions=math.isqrt(num_drones))
            sim_b.start()

            results['boustrophedon'][num_drones] = sim_b.step_count
            print(f"  Boustrophedon: {sim_b.step_count} steps")
            
            # Inverse ACO
            sim_i = InverseACOSimulation(grid_size=100, num_drones=num_drones)
            sim_i.start()

            results['inverse_aco'][num_drones] = sim_i.step_count
            print(f"  Inverse ACO: {sim_i.step_count} steps")
        
        self.results['scalability_drones'] = results
        self._save_results('scalability_drones')
        self._plot_scalability_drones(results)
    
    def run_scalability_grid(self):
        """Experiment 4: Scalability (Grid Size)"""
        print("\n" + "="*60)
        print("EXPERIMENT 4: Scalability (Grid Size)")
        print("="*60)
        
        configs = [
            (50, 9, 3),    # Small
            (100, 25, 5),  # Medium
            (150, 49, 7)   # Large
        ]
        
        results = {'boustrophedon': {}, 'inverse_aco': {}}
        
        for grid_size, num_drones, cell_div in configs:
            print(f"\nTesting {grid_size}×{grid_size} grid with {num_drones} drones...")
            
            # Boustrophedon
            sim_b = BoustrophedonSimulation(grid_size=grid_size, num_drones=num_drones, cell_divisions=cell_div)
            sim_b.start()

            results['boustrophedon'][grid_size] = sim_b.step_count
            print(f"  Boustrophedon: {sim_b.step_count} steps")
            
            # Inverse ACO
            sim_i = InverseACOSimulation(grid_size=grid_size, num_drones=num_drones)
            sim_i.start()
            
            results['inverse_aco'][grid_size] = sim_i.step_count
            print(f"  Inverse ACO: {sim_i.step_count} steps")
        
        self.results['scalability_grid'] = results
        self._save_results('scalability_grid')
        self._plot_scalability_grid(results)
    
    def run_parameter_sweep(self):
        """Experiment 5: Parameter Sensitivity (IACO only)"""
        print("\n" + "="*60)
        print("EXPERIMENT 5: Parameter Sensitivity")
        print("="*60)
        
        alphas = [1, 2, 3, 5]
        betas = [1, 2, 3, 5]
        
        results = np.zeros((len(alphas), len(betas)))
        
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                print(f"\nTesting α={alpha}, β={beta}...")
                sim = InverseACOSimulation(grid_size=100, num_drones=25, alpha=alpha, beta=beta)
                sim.start()

                results[i, j] = sim.step_count
                print(f"  Steps: {sim.step_count}")
        
        self.results['parameter_sweep'] = {'alphas': alphas, 'betas': betas, 'results': results.tolist()}
        self._save_results('parameter_sweep')
        self._plot_parameter_sweep(alphas, betas, results)
    
    def _save_results(self, experiment_name):
        """Save results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/data/{experiment_name}_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results.get(experiment_name, {}), f, indent=2)
        print(f"\nResults saved to {filename}")
    
    def _plot_baseline(self, results):
        """Plot baseline comparison"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Steps comparison
        b_steps = [r['steps'] for r in results['boustrophedon']]
        i_steps = [r['steps'] for r in results['inverse_aco']]
        
        ax1.bar(['Boustrophedon', 'Inverse ACO'], 
                [np.mean(b_steps), np.mean(i_steps)],
                yerr=[np.std(b_steps), np.std(i_steps)],
                capsize=5, color=['#1f77b4', '#ff7f0e'])
        ax1.set_ylabel('Steps to 95% Coverage')
        ax1.set_title('Baseline Performance')
        ax1.grid(True, alpha=0.3)
        
        # Redundancy comparison
        b_red = [r['redundancy'] for r in results['boustrophedon']]
        i_red = [r['redundancy'] for r in results['inverse_aco']]
        
        ax2.bar(['Boustrophedon', 'Inverse ACO'],
                [np.mean(b_red), np.mean(i_red)],
                yerr=[np.std(b_red), np.std(i_red)],
                capsize=5, color=['#1f77b4', '#ff7f0e'])
        ax2.set_ylabel('Redundancy Factor')
        ax2.set_title('Path Efficiency')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/plots/baseline.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to {self.output_dir}/plots/baseline.png")
        plt.close()
    
    def _plot_scalability_drones(self, results):
        """Plot scalability vs drone count"""
        plt.figure(figsize=(10, 6))
        
        drones = sorted(results['boustrophedon'].keys())
        b_steps = [results['boustrophedon'][d] for d in drones]
        i_steps = [results['inverse_aco'][d] for d in drones]
        
        plt.plot(drones, b_steps, 'o-', linewidth=2, markersize=8, label='Boustrophedon')
        plt.plot(drones, i_steps, 's-', linewidth=2, markersize=8, label='Inverse ACO')
        
        plt.xlabel('Number of Drones')
        plt.ylabel('Steps to 95% Coverage')
        plt.title('Scalability: Drone Count')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f'{self.output_dir}/plots/scalability_drones.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to {self.output_dir}/plots/scalability_drones.png")
        plt.close()
    
    def _plot_scalability_grid(self, results):
        """Plot scalability vs grid size"""
        plt.figure(figsize=(10, 6))
        
        grids = sorted(results['boustrophedon'].keys())
        b_steps = [results['boustrophedon'][g] for g in grids]
        i_steps = [results['inverse_aco'][g] for g in grids]
        
        plt.plot(grids, b_steps, 'o-', linewidth=2, markersize=8, label='Boustrophedon')
        plt.plot(grids, i_steps, 's-', linewidth=2, markersize=8, label='Inverse ACO')
        
        plt.xlabel('Grid Size')
        plt.ylabel('Steps to 95% Coverage')
        plt.title('Scalability: Grid Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(f'{self.output_dir}/plots/scalability_grid.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to {self.output_dir}/plots/scalability_grid.png")
        plt.close()
    
    def _plot_parameter_sweep(self, alphas, betas, results):
        """Plot parameter sensitivity heatmap"""
        plt.figure(figsize=(10, 8))
        
        im = plt.imshow(results, cmap='viridis', aspect='auto')
        plt.colorbar(im, label='Steps to 95% Coverage')
        
        plt.xticks(range(len(betas)), betas)
        plt.yticks(range(len(alphas)), alphas)
        plt.xlabel('β (Heuristic Weight)')
        plt.ylabel('α (Pheromone Weight)')
        plt.title('IACO Parameter Sensitivity')
        
        # Annotate cells with values
        for i in range(len(alphas)):
            for j in range(len(betas)):
                plt.text(j, i, f'{int(results[i, j])}',
                        ha='center', va='center', color='white', fontweight='bold')
        
        plt.savefig(f'{self.output_dir}/plots/parameter_sweep.png', dpi=300, bbox_inches='tight')
        print(f"Plot saved to {self.output_dir}/plots/parameter_sweep.png")
        plt.close()
    
    def run_all(self):
        """Run all experiments"""
        import os
        os.makedirs(f'{self.output_dir}/plots', exist_ok=True)
        os.makedirs(f'{self.output_dir}/data', exist_ok=True)
        
        print("\n" + "="*60)
        print("RUNNING ALL EXPERIMENTS")
        print("="*60)
        
        # self.run_baseline(trials=5)       #DONE
        self.run_scalability_drones()
        self.run_scalability_grid()
        self.run_parameter_sweep()
        
        print("\n" + "="*60)
        print("ALL EXPERIMENTS COMPLETE!")
        print("="*60)


if __name__ == "__main__":
    runner = ExperimentRunner()
    runner.run_all()