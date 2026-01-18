import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import os

class InverseACOSimulation:
    def __init__(self, grid_size=15, num_drones=9, eps=1e-8, pheromone_radius=1, alpha=2, beta=3, evaporation_rate=0.005,
                 failure_threshold=-1):
        self.grid_size = grid_size
        self.num_drones = num_drones
        self.active_drone_count = 0
        self.launch_interval = 5
        self.pheromone_radius = pheromone_radius #drone leaves pheromone in either one tile (radius=0) or multiple tiles (on a radius in front of it)
        self.failure_threshold = failure_threshold
        
        self.alpha = alpha  # repulsion strength (pheromone influence)
        self.beta = beta   # exploration drive (heuristic influence)
        self.evaporation_rate = evaporation_rate
        self.Q = 1  # pheromone deposit constant
        self.eps = eps
        
        
        self.coverage_grid = np.zeros((grid_size, grid_size))  # Not seen by drones; logs performance
        self.pheromone_map = np.zeros((grid_size, grid_size))

        # Tracks if an individual drone has visited a tile (1.0 for yes, 0.0 for no)
        self.personal_maps = np.zeros((num_drones, grid_size, grid_size))

        # Statistics
        self.step_count = 0
        self.coverage = 0.0
        self.redundancy = 0.0


        # Edges pheromone initialization (keeps the drones away from starting corner; similar to the IACA paper)
        max_pheromone = self.Q * self.num_drones
        lam = 0.9
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # Find the distance to the closest edge
                dist_to_edge = min(i, self.grid_size - 1 - i, j, self.grid_size - 1 - j)
            
                # Pheromone is highest at distance 0 (the edge)
                self.pheromone_map[i, j] = max_pheromone * (lam ** dist_to_edge)

        # All drones start in a corner
        start = 0
        self.drones = np.full((num_drones, 2), start, dtype=int)
        
        # Initialize pheromone at starting position
        self.pheromone_map[start, start] = self.Q * num_drones
        self.coverage_grid[start, start] = 0.0

    def compute_heuristic(self, x, y):
        """
        FROM THE IACA PAPER
        Priority-based Heuristic: High sensitivity to unvisited areas 
        using the paper's logarithmic scale.
        """
        local_sum_priority = 0
        cells_checked = 0
        radius = 1 
        
        MAX_P = self.Q * self.num_drones

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    # Normalize the pheromone value
                    p_val = self.pheromone_map[nx, ny]
                    p_norm = np.clip(p_val / MAX_P, self.eps, 1.0)
                    
                    # Compute Logarithmic Priority
                    # Priority is high (attractive) when p_norm is low (clean)
                    priority = 1.0 - np.log2(p_norm)
                    
                    local_sum_priority += priority
                    cells_checked += 1
        
        # Return average priority of the neighborhood
        return local_sum_priority / cells_checked if cells_checked > 0 else 1.0

    def roulette_wheel_selection(self, neighbors, probabilities):
        """Select a neighbor based on probability distribution."""
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        
        for i, threshold in enumerate(cumulative_probabilities):
            if r <= threshold:
                return neighbors[i]
        
        return neighbors[-1]  # Fallback to last neighbor

    def get_move(self, drone_idx):
        x, y = self.drones[drone_idx]
        
        # Neighborhood Check
        clean_neighbors = 0
        threshold = 0.5 
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    if self.pheromone_map[nx, ny] < threshold:
                        clean_neighbors += 1

        probabilities = []
        neighbors = []

        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                tau = self.pheromone_map[nx, ny]
                inv_pheromone = 1.0 / (1.0 + tau)
                eta = self.compute_heuristic(nx, ny)
                
                # Base IACO Score
                score = (inv_pheromone ** self.alpha) * (eta ** self.beta)

                # PERSONAL MEMORY PENALTY
                # If THIS drone has been here before, heavily discourage returning
                if self.personal_maps[drone_idx, nx, ny] > 0:
                    score *= 0.01 # Hard penalty to prevent self-overlap
                
                probabilities.append(score)
                neighbors.append((nx, ny, dx, dy))
        
        # Selection and History Update
        prob_sum = sum(probabilities)
        # Handle edge case where all moves are heavily penalized
        norm_probs = [p/prob_sum for p in probabilities] if prob_sum > 0 else [1/len(neighbors)]*len(neighbors)
        
        choice = self.roulette_wheel_selection(neighbors, norm_probs)
        
        # Mark this tile in the personal map so drones don't come back
        self.personal_maps[drone_idx, choice[0], choice[1]] = 1.0
        
        return (choice[0], choice[1])

    def deposit_pheromone(self, x, y, radius=2):
        """
        Spreads pheromone in a radius around the drone's position.
        The intensity decreases as it gets further from the center.
        """
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                    dist = np.sqrt(dx**2 + dy**2)
                    # Amount decreases with distance
                    deposit_amount = self.Q / (1.0 + dist)
                    self.pheromone_map[nx, ny] += deposit_amount


    def step(self):
        # 1. Increment active drones based on interval
        if self.step_count % self.launch_interval == 0 and self.active_drone_count < self.num_drones and self.step_count < 400:
            self.active_drone_count += 1

        self.step_count += 1

        if self.step_count == self.failure_threshold:
            reduction = self.active_drone_count // 2
            self.active_drone_count -= reduction
            print(f"!!! CRITICAL FAILURE: {reduction} drones lost at step {self.step_count} !!!")
        
        # 2. Only move drones that have been 'launched'
        for i in range(self.active_drone_count):
            new_pos = self.get_move(i)
            self.drones[i] = new_pos

            # Deposit in a radius instead of just one tile
            self.deposit_pheromone(new_pos[0], new_pos[1], radius=self.pheromone_radius)

            self.coverage_grid[new_pos[0], new_pos[1]] += 1.0

        self.pheromone_map *= (1 - self.evaporation_rate)

    def start(self):
        animate_swarm(self, steps=200, show_divisions=True, divisions=15)

def save_results_to_csv(sim, filename="aco_experiments.csv"):    
    # Data row
    data = {
        "grid_size": sim.grid_size,
        "num_drones": sim.num_drones,
        "alpha": sim.alpha,
        "beta": sim.beta,
        "evaporation": sim.evaporation_rate,
        "total_steps": sim.step_count,
        "final_coverage": round(sim.coverage, 2),
        "redundancy_factor": round(sim.redundancy, 3)
    }
    
    file_exists = os.path.isfile(filename)
    with open(filename, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(data)
    print(f"Results saved to {filename}")
    

def animate_swarm(sim, steps=20000, show_divisions=True, divisions=3):
    """Animate the swarm exploration with optional grid divisions for visualization."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    grid_size = sim.grid_size
    
    im1 = ax1.imshow(sim.coverage_grid, cmap='Greens', origin='lower', 
                     extent=[0, grid_size, 0, grid_size], vmin=0, vmax=5)
    im2 = ax2.imshow(sim.pheromone_map, cmap='hot', origin='lower', 
                     extent=[0, grid_size, 0, grid_size], vmin=0, vmax=10)
    
    ax1.set_title('Coverage Grid (Recent Visits)', fontsize=12, fontweight='bold')
    ax2.set_title('Pheromone Map (Repulsion)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    
    # Add colorbars
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Optionally draw reference grid divisions (for visualization only)
    if show_divisions and divisions > 1:
        step = grid_size / divisions
        for i in range(1, divisions):
            coord = i * step
            for ax in [ax1, ax2]:
                ax.axvline(x=coord, color='gray', linestyle='--', linewidth=1, alpha=0.3)
                ax.axhline(y=coord, color='gray', linestyle='--', linewidth=1, alpha=0.3)
    
    # Stats text positioned at the top of the left subplot
    stats_text = ax1.text(grid_size/2, grid_size * 1.08, "", 
                          ha='center', fontsize=11, fontweight='bold',
                          transform=ax1.transData)
    
    # Drone visualization with unique colors
    colors = plt.cm.tab20(np.linspace(0, 1, sim.num_drones))
    
    # Initialize scatter with actual drone positions
    drone_scatter = ax1.scatter(sim.drones[:, 1] + 0.5, sim.drones[:, 0] + 0.5, 
                                c=colors, s=100, marker='o', 
                                edgecolors='white', linewidths=2, zorder=5)

    def update(frame):
        # Prevent errors if the figure is closed
        if not plt.fignum_exists(fig.number):
            return
        
        sim.step()
        
        # Calculate unique coverage: how many tiles have been visited at least once
        unique_visited = np.sum(sim.coverage_grid > 0)
        total_tiles = sim.grid_size ** 2
        coverage_pct = (unique_visited / total_tiles) * 100
        
        # Update Visuals
        im1.set_data(sim.coverage_grid)
        im2.set_data(sim.pheromone_map)
        drone_scatter.set_offsets(np.c_[sim.drones[:, 1] + 0.5, sim.drones[:, 0] + 0.5])
        
        # Display Stats
        stats_text.set_text(f"Step: {sim.step_count} | Coverage: {coverage_pct:.2f}%")
        
        # STOP CRITERION: 95%
        if coverage_pct >= 95.0:
            final_steps = sim.step_count
            stats_text.set_text(f"SEARCH COMPLETE (95%) | Step: {final_steps}")
            stats_text.set_color('blue')
            ani.event_source.stop() # Stop the animation timer

            unique_visited = np.sum(sim.coverage_grid > 0)
            total_tiles = sim.grid_size ** 2
            coverage_pct = (unique_visited / total_tiles) * 100
    
            total_visits = np.sum(sim.coverage_grid)
            redundancy_factor = total_visits / unique_visited if unique_visited > 0 else 0

            sim.redundancy = redundancy_factor
            sim.coverage = coverage_pct

            save_results_to_csv(sim)
            
        return im1, im2, drone_scatter, stats_text

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    
    ani = FuncAnimation(fig, update, frames=steps, interval=100, blit=False)
    plt.show()


if __name__ == "__main__":
    sim = InverseACOSimulation(grid_size=100, num_drones=30, pheromone_radius=2, alpha=7, beta=2, evaporation_rate=0.005)
    # animate_swarm(sim, steps=200, show_divisions=True, divisions=15)
    sim.start()