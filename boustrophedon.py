import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class BoustrophedonSimulation:
    def __init__(self, grid_size=15, num_drones=9, cell_divisions=3, failure_threshold=-1):
        self.grid_size = grid_size
        self.num_drones = num_drones
        self.cell_divisions = cell_divisions

        self.failure_threshold = failure_threshold
        
        # Cell decomposition
        self.cell_size = grid_size // cell_divisions
        self.total_cells = cell_divisions * cell_divisions
        
        # Coverage tracking
        self.coverage_grid = np.zeros((grid_size, grid_size))
        
        # Drone states
        center = 0
        self.drones = np.full((num_drones, 2), center, dtype=float)
        self.drone_cells = [-1] * num_drones  # Which cell each drone is assigned to
        self.drone_directions = np.ones(num_drones, dtype=int)  # 1 = right, -1 = left
        self.drone_target_row = np.zeros(num_drones, dtype=int)  # Target row in cell
        self.drone_state = ['seeking'] * num_drones  # 'seeking', 'sweeping', 'done'
        
        # Drone failure tracking
        self.drone_active = np.ones(num_drones, dtype=bool)  # Track which drones are operational
        self.active_drone_count = num_drones
        
        # Statistics
        self.step_count = 0
        self.coverage = 0.0
        self.redundancy = 0.0
        
        # Initialize
        self.coverage_grid[center, center] = 1.0
        self.assign_cells()  # Initial assignment
    
    def get_cell_bounds(self, cell_id):
        """Get the bounds of a cell given its ID (0 to total_cells-1)."""
        row = cell_id // self.cell_divisions
        col = cell_id % self.cell_divisions
        
        x_min = col * self.cell_size
        x_max = min((col + 1) * self.cell_size, self.grid_size)
        y_min = row * self.cell_size
        y_max = min((row + 1) * self.cell_size, self.grid_size)
        
        return x_min, x_max, y_min, y_max
    
    def get_coverage(self):
        return
    
    def get_cell_coverage_score(self, cell_id):
        """Calculate how much of a cell is uncovered (0-1, higher = less covered)."""
        if cell_id < 0 or cell_id >= self.total_cells:
            return 0
        
        x_min, x_max, y_min, y_max = self.get_cell_bounds(cell_id)
        cell_area = self.coverage_grid[y_min:y_max, x_min:x_max]
        
        # Return fraction of uncovered area
        total_cells = cell_area.size
        covered_cells = np.sum(cell_area > 0.1)
        return (total_cells - covered_cells) / total_cells if total_cells > 0 else 0
    
    def assign_cells(self):
        """Assign drones to cells - one drone per cell."""
        # Calculate scores for each cell (how much it needs coverage)
        cell_scores = []
        for cell_id in range(self.total_cells):
            score = self.get_cell_coverage_score(cell_id)
            cell_scores.append((cell_id, score))
        
        # Sort cells by coverage need (descending)
        cell_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Assign drones to cells (one per cell) - only consider ACTIVE drones
        unassigned_drones = [i for i in range(self.num_drones) if self.drone_active[i]]
        
        # First pass: Keep drones in their current cells if they're not done and cell still needs work
        for drone_idx in list(unassigned_drones):
            current_cell = self.drone_cells[drone_idx]
            if current_cell >= 0 and self.drone_state[drone_idx] in ['seeking', 'sweeping']:
                # Check if this cell still needs coverage
                if self.get_cell_coverage_score(current_cell) > 0.1:
                    # Keep this drone in its current cell
                    unassigned_drones.remove(drone_idx)
        
        # Second pass: Assign remaining drones to uncovered cells
        for cell_id, score in cell_scores:
            if not unassigned_drones:
                break
            
            # Skip if cell is already assigned to an active drone
            if any(self.drone_cells[i] == cell_id and self.drone_active[i] 
                   for i in range(self.num_drones)):
                continue
            
            # Find nearest unassigned drone to this cell
            x_min, x_max, y_min, y_max = self.get_cell_bounds(cell_id)
            cell_center_y = (y_min + y_max) / 2
            cell_center_x = (x_min + x_max) / 2
            
            best_drone = None
            best_distance = float('inf')
            
            for drone_idx in unassigned_drones:
                y, x = self.drones[drone_idx]
                distance = np.sqrt((cell_center_y - y)**2 + (cell_center_x - x)**2)
                
                if distance < best_distance:
                    best_distance = distance
                    best_drone = drone_idx
            
            if best_drone is not None:
                self.drone_cells[best_drone] = cell_id
                unassigned_drones.remove(best_drone)
                
                # Initialize lawnmower pattern for this drone
                self.drone_target_row[best_drone] = y_min
                self.drone_state[best_drone] = 'seeking'
    
    def lawnmower_move(self, drone_idx):
        """Execute lawnmower pattern within assigned cell."""
        cell_id = self.drone_cells[drone_idx]
        if cell_id < 0:
            return np.array([0.0, 0.0])
        
        x_min, x_max, y_min, y_max = self.get_cell_bounds(cell_id)
        y, x = self.drones[drone_idx]
        
        # Current grid position
        grid_y, grid_x = int(round(y)), int(round(x))
        
        state = self.drone_state[drone_idx]
        
        # move to start of assigned cell
        if state == 'seeking':
            if y_min <= grid_y < y_max and x_min <= grid_x < x_max:
                self.drone_state[drone_idx] = 'sweeping'
                self.drone_target_row[drone_idx] = y_min
                self.drone_directions[drone_idx] = 1
            else:
                # Move toward cell's starting corner (bottom-left)
                target_y = y_min + 0.5
                target_x = x_min + 0.5
                diff = np.array([target_y - y, target_x - x])
                dist = np.linalg.norm(diff)
                return diff / dist if dist > 0.1 else np.array([0.0, 0.0])
        
        # sweeping
        elif state == 'sweeping':
            target_row = self.drone_target_row[drone_idx]
            direction = self.drone_directions[drone_idx]
            
            if abs(grid_y - target_row) > 0:
                return np.array([1.0 if grid_y < target_row else -1.0, 0.0])
            
            target_x = grid_x + direction
            
            if target_x <= x_min or target_x > x_max:
                self.drone_directions[drone_idx] *= -1
                if target_row + 1 < y_max:
                    self.drone_target_row[drone_idx] = target_row + 1
                    return np.array([1.0, 0.0])
                else:
                    self.drone_state[drone_idx] = 'done'
                    return np.array([0.0, 0.0])
            
            return np.array([0.0, float(direction)])
        else:
            return np.array([0.0, 0.0])
        
        return np.array([0.0, 0.0])
    
    def step(self):
        """Execute one simulation step."""
        self.step_count += 1
        
        # CRITICAL FAILURE
        if self.step_count == self.failure_threshold:
            reduction = self.active_drone_count // 2
            # Deactivate half the drones
            active_indices = [i for i in range(self.num_drones) if self.drone_active[i]]
            drones_to_fail = np.random.choice(active_indices, size=reduction, replace=False)
            
            for drone_idx in drones_to_fail:
                self.drone_active[drone_idx] = False
                self.drone_cells[drone_idx] = -1  # Unassign cell
                self.drone_state[drone_idx] = 'failed'
            
            self.active_drone_count -= reduction
            print(f"!!! CRITICAL FAILURE: {reduction} drones lost at step {self.step_count} !!!")
            
            # Immediately reassign cells to remaining drones
            self.assign_cells()
        
        # Reassign cells if many drones are done
        done_drones = sum(1 for i, state in enumerate(self.drone_state) 
                         if state == 'done' and self.drone_active[i])
        active_drones = sum(1 for active in self.drone_active if active)
        
        if active_drones > 0 and done_drones > active_drones // 2:
            self.assign_cells()
        
        step_distance = 0
        
        for i in range(self.num_drones):
            if not self.drone_active[i]:
                continue

            old_pos = self.drones[i].copy()
            movement = self.lawnmower_move(i)
            
            if np.linalg.norm(movement) > 0:
                movement = movement / np.linalg.norm(movement)
            
            new_pos = self.drones[i] + movement
            # Clamp to grid bounds
            new_pos[0] = np.clip(new_pos[0], 0, self.grid_size - 0.01)
            new_pos[1] = np.clip(new_pos[1], 0, self.grid_size - 0.01)
            
            self.drones[i] = new_pos
            
            # Track distance
            distance = np.linalg.norm(new_pos - old_pos)
            step_distance += distance
            
            # Mark coverage on grid
            grid_y, grid_x = int(round(new_pos[0])), int(round(new_pos[1]))
            if 0 <= grid_y < self.grid_size and 0 <= grid_x < self.grid_size:
                self.coverage_grid[grid_y, grid_x] += 1.0

    def start(self):
        animate_swarm(self, steps=300)


def animate_swarm(sim, steps=300):
    """Animate the Boustrophedon simulation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    grid_size = sim.grid_size
    # Coverage grid
    im1 = ax1.imshow(sim.coverage_grid, cmap='Greens', origin='lower',
                     extent=[0, grid_size, 0, grid_size], vmin=0, vmax=10)
    
    # Cell assignment visualization
    cell_assignment_grid = np.zeros((grid_size, grid_size))
    im2 = ax2.imshow(cell_assignment_grid, cmap='tab20', origin='lower',
                     extent=[0, grid_size, 0, grid_size], vmin=0, vmax=20)
    
    ax1.set_title('Coverage Grid', fontsize=12, fontweight='bold')
    ax2.set_title('Cell Assignments (Boustrophedon)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax2.set_xlabel('X Position')
    ax2.set_ylabel('Y Position')
    
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    #plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Draw cell boundaries
    step = grid_size / sim.cell_divisions
    for i in range(1, sim.cell_divisions):
        coord = i * step
        for ax in [ax1, ax2]:
            ax.axvline(x=coord, color='blue', linestyle='-', linewidth=2, alpha=0.6)
            ax.axhline(y=coord, color='blue', linestyle='-', linewidth=2, alpha=0.6)
    
    # Stats text
    stats_text = ax1.text(grid_size/2, grid_size * 1.08, "",
                          ha='center', fontsize=10, fontweight='bold',
                          transform=ax1.transData)
    
    # Drone visualization
    colors = plt.cm.tab20(np.linspace(0, 1, sim.num_drones))
    
    drone_scatter1 = ax1.scatter(sim.drones[:, 1], sim.drones[:, 0],
                                 c=colors, s=100, marker='o',
                                 edgecolors='white', linewidths=2, zorder=5)
    drone_scatter2 = ax2.scatter(sim.drones[:, 1], sim.drones[:, 0],
                                 c=colors, s=100, marker='o',
                                 edgecolors='white', linewidths=2, zorder=5)

    def update(frame):
        sim.step()
        im1.set_data(sim.coverage_grid)
        
        # Create cell assignment visualization
        cell_assignment_grid = np.zeros((grid_size, grid_size))
        for drone_idx in range(sim.num_drones):
            cell_id = sim.drone_cells[drone_idx]
            if cell_id >= 0:
                x_min, x_max, y_min, y_max = sim.get_cell_bounds(cell_id)
                cell_assignment_grid[y_min:y_max, x_min:x_max] = drone_idx + 1
        
        im2.set_data(cell_assignment_grid)
        
        # Update drone positions (only show active drones)
        active_positions = np.array([sim.drones[i] for i in range(sim.num_drones) 
                                     if sim.drone_active[i]])
        active_colors = [colors[i] for i in range(sim.num_drones) if sim.drone_active[i]]
        
        if len(active_positions) > 0:
            drone_scatter1.set_offsets(np.c_[active_positions[:, 1], active_positions[:, 0]])
            drone_scatter1.set_color(active_colors)
            drone_scatter2.set_offsets(np.c_[active_positions[:, 1], active_positions[:, 0]])
            drone_scatter2.set_color(active_colors)
        else:
            # No active drones
            drone_scatter1.set_offsets(np.empty((0, 2)))
            drone_scatter2.set_offsets(np.empty((0, 2)))
        
        # Statistics
        total_coverage = np.sum(sim.coverage_grid > 0.1) / (grid_size**2) * 100
        
        # Count assigned drones (only active ones)
        assigned_drones = sum(1 for i, cell in enumerate(sim.drone_cells) 
                            if cell >= 0 and sim.drone_active[i])
        active_drones = sum(1 for active in sim.drone_active if active)

        # Calculate unique coverage: how many tiles have been visited at least once
        unique_visited = np.sum(sim.coverage_grid > 0)
        total_tiles = sim.grid_size ** 2
        coverage_pct = (unique_visited / total_tiles) * 100
        
        stats_text.set_text(
            f"Step: {sim.step_count} | Coverage: {total_coverage:.1f}% | "
            f"Active: {active_drones}/{sim.num_drones} | Assigned: {assigned_drones}"
        )

        if coverage_pct >= 95.0:
            final_steps = sim.step_count
            stats_text.set_text(f"SEARCH COMPLETE (95%) | Step: {final_steps}")
            stats_text.set_color('blue')
            ani.event_source.stop()

            unique_visited = np.sum(sim.coverage_grid > 0)
            total_tiles = sim.grid_size ** 2
            coverage_pct = (unique_visited / total_tiles) * 100
    
            total_visits = np.sum(sim.coverage_grid)
            redundancy_factor = total_visits / unique_visited if unique_visited > 0 else 0

            print(f"Unique visited: {unique_visited}")
            print(f"Total tiles: {total_tiles}")
            print(f"Coverage %: {coverage_pct:.2f}%")
            print(f"Total visits: {total_visits:.0f}")
            print(f"Redundancy factor: {redundancy_factor:.2f}")

            sim.redundancy = redundancy_factor
            sim.coverage = coverage_pct
        
        return [im1, im2, drone_scatter1, drone_scatter2, stats_text]

    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    
    ani = FuncAnimation(fig, update, frames=steps, interval=100, blit=False)
    plt.show()


if __name__ == "__main__":
    sim = BoustrophedonSimulation(grid_size=100, num_drones=21, cell_divisions=5)
    # animate_swarm(sim, steps=300)
    sim.start()