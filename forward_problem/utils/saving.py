import numpy as np
import os
import pickle
from datetime import datetime

def save_data(history_phix, history_phiy, history_thetax, history_thetay, Laser_coords, example_name=None):
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if example_name:
        folder_name = f"{timestamp}_{example_name}"
    else:
        folder_name = f"{timestamp}_simulation"
    
    output_directory = os.path.join("output", "examples", folder_name)
    os.makedirs(output_directory, exist_ok=True)  # Ensure the directory exists
    filepath = os.path.join(output_directory, "simulation_data.pkl")

    # Collect all data into a dictionary
    data = {
        "history_phix": history_phix,
        "history_phiy": history_phiy,
        "history_thetax": history_thetax,
        "history_thetay": history_thetay,
        "Laser_coords" : Laser_coords
    }

    # Write the data to a file using pickle
    with open(filepath, 'wb') as file:
        pickle.dump(data, file)

    print(f"Data saved to {filepath}")
    
    # Also save workpiece projection data as CSV
    save_workpiece_projection(Laser_coords, output_directory)

def save_workpiece_projection(Laser_coords, output_directory):
    """Save workpiece laser projection data as CSV and summary text file."""
    import csv
    
    # Extract workpiece projections
    workpiece_projections = []
    for idx, coords in Laser_coords.items():
        if coords:  # Make sure coords is not empty
            final_pos = coords[-1]  # Last position (at workpiece)
            time = idx * 0.1  # Convert index to time
            x, y = final_pos[0], final_pos[1]
            distance = np.sqrt(x**2 + y**2)
            workpiece_projections.append([time, x, y, distance])
    
    if workpiece_projections:
        # Save as CSV
        csv_filepath = os.path.join(output_directory, "workpiece_projections.csv")
        with open(csv_filepath, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time_s', 'X_Position', 'Y_Position', 'Distance_from_Center'])
            writer.writerows(workpiece_projections)
        
        # Calculate statistics
        x_vals = [row[1] for row in workpiece_projections]
        y_vals = [row[2] for row in workpiece_projections]
        distances = [row[3] for row in workpiece_projections]
        
        x_range = max(x_vals) - min(x_vals)
        y_range = max(y_vals) - min(y_vals)
        scan_area = x_range * y_range
        center_x, center_y = np.mean(x_vals), np.mean(y_vals)
        effective_diameter = 2 * np.percentile(distances, 95)
        
        # Save summary text file
        summary_filepath = os.path.join(output_directory, "workpiece_analysis.txt")
        with open(summary_filepath, 'w') as f:
            f.write("RISLEY PRISM WORKPIECE PROJECTION ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total scan positions: {len(workpiece_projections)}\n")
            f.write(f"Time range: 0.0 to {workpiece_projections[-1][0]:.1f} seconds\n\n")
            f.write("SCAN AREA STATISTICS:\n")
            f.write(f"  X range: {x_range:.3f} units\n")
            f.write(f"  Y range: {y_range:.3f} units\n")
            f.write(f"  Total scan area: {scan_area:.3f} square units\n")
            f.write(f"  Center position: ({center_x:.3f}, {center_y:.3f})\n")
            f.write(f"  Effective diameter (95%): {effective_diameter:.3f} units\n")
            f.write(f"  Maximum displacement: {max(distances):.3f} units\n")
            f.write(f"  Mean displacement: {np.mean(distances):.3f} units\n")
            f.write(f"  Standard deviation: {np.std(distances):.3f} units\n\n")
            f.write("POSITION EXTREMES:\n")
            f.write(f"  Min X: {min(x_vals):.3f}, Max X: {max(x_vals):.3f}\n")
            f.write(f"  Min Y: {min(y_vals):.3f}, Max Y: {max(y_vals):.3f}\n")
        
        print(f"Workpiece projection data saved to {csv_filepath}")
        print(f"Workpiece analysis summary saved to {summary_filepath}")
        
        # Generate and save workpiece visualization images
        save_workpiece_images(workpiece_projections, output_directory)

def save_workpiece_images(workpiece_projections, output_directory):
    """Generate and save workpiece visualization images."""
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    import matplotlib.colors as colors
    from matplotlib.patches import Circle
    
    if not workpiece_projections:
        return
    
    # Extract data
    times = [row[0] for row in workpiece_projections]
    x_vals = [row[1] for row in workpiece_projections]
    y_vals = [row[2] for row in workpiece_projections]
    distances = [row[3] for row in workpiece_projections]
    
    # Calculate statistics for visualization
    center_x, center_y = np.mean(x_vals), np.mean(y_vals)
    effective_diameter = 2 * np.percentile(distances, 95)
    
    # Create the main workpiece projection plot
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Risley Prism Workpiece Laser Projection Analysis', fontsize=16, weight='bold')
    
    # Plot 1: 2D Workpiece Projection with Time Color Coding
    cmap = cm.get_cmap('viridis')
    norm = colors.Normalize(vmin=min(times), vmax=max(times))
    
    # Calculate appropriate plot bounds with padding for better visibility
    x_range = max(x_vals) - min(x_vals)
    y_range = max(y_vals) - min(y_vals)
    padding_factor = 0.25  # 25% padding around the pattern
    
    x_padding = max(x_range * padding_factor, 0.15)  # Minimum 0.15 units padding
    y_padding = max(y_range * padding_factor, 0.15)
    
    x_min = min(x_vals) - x_padding
    x_max = max(x_vals) + x_padding
    y_min = min(y_vals) - y_padding
    y_max = max(y_vals) + y_padding
    
    # Dynamic point sizing for analysis plot
    analysis_point_size = max(10, min(50, 2000 // len(x_vals))) 
    scatter = ax1.scatter(x_vals, y_vals, c=times, cmap=cmap, s=analysis_point_size, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    # Add 95% containment circle
    circle = Circle((center_x, center_y), effective_diameter/2, 
                   fill=False, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
    ax1.add_patch(circle)
    
    # Mark center
    ax1.plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=3)
    
    # Connect points to show scan path - thicker for visibility
    ax1.plot(x_vals, y_vals, 'k-', alpha=0.5, linewidth=1.5)
    
    # Set tight bounds for better zoom on the pattern
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    
    ax1.set_xlabel('X Position (units)', fontsize=12, weight='bold')
    ax1.set_ylabel('Y Position (units)', fontsize=12, weight='bold')
    ax1.set_title('Workpiece Laser Projection Pattern', fontsize=14, weight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Time (seconds)', fontsize=11, weight='bold')
    
    # Add statistics text box
    stats_text = f'Scan Area: {(max(x_vals)-min(x_vals))*(max(y_vals)-min(y_vals)):.2f} sq units\n'
    stats_text += f'Center: ({center_x:.2f}, {center_y:.2f})\n'
    stats_text += f'Effective Ø: {effective_diameter:.2f} units\n'
    stats_text += f'Points: {len(workpiece_projections)}'
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            verticalalignment='top', fontsize=10)
    
    # Plot 2: X and Y Position vs Time
    ax2.plot(times, x_vals, 'b-', label='X Position', linewidth=2, alpha=0.8)
    ax2.plot(times, y_vals, 'r-', label='Y Position', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Time (seconds)', fontsize=12, weight='bold')
    ax2.set_ylabel('Position (units)', fontsize=12, weight='bold')
    ax2.set_title('Beam Position vs Time', fontsize=14, weight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Distance from Center vs Time
    ax3.plot(times, distances, 'g-', linewidth=2, alpha=0.8)
    ax3.axhline(y=effective_diameter/2, color='r', linestyle='--', linewidth=2,
               label=f'95% radius: {effective_diameter/2:.2f}')
    ax3.axhline(y=np.mean(distances), color='orange', linestyle=':', linewidth=2,
               label=f'Mean: {np.mean(distances):.2f}')
    ax3.set_xlabel('Time (seconds)', fontsize=12, weight='bold')
    ax3.set_ylabel('Distance from Center (units)', fontsize=12, weight='bold')
    ax3.set_title('Beam Displacement from Center', fontsize=14, weight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Histogram of Positions
    ax4.hist2d(x_vals, y_vals, bins=15, cmap='Blues', alpha=0.7)
    
    # Set tight bounds for better zoom on the pattern (same as ax1)
    ax4.set_xlim(x_min, x_max)
    ax4.set_ylim(y_min, y_max)
    
    ax4.set_xlabel('X Position (units)', fontsize=12, weight='bold')
    ax4.set_ylabel('Y Position (units)', fontsize=12, weight='bold')
    ax4.set_title('Beam Position Density', fontsize=14, weight='bold')
    ax4.set_aspect('equal')
    
    # Add the same circle and center to histogram
    circle2 = Circle((center_x, center_y), effective_diameter/2, 
                    fill=False, color='red', linestyle='--', linewidth=2.5, alpha=0.8)
    ax4.add_patch(circle2)
    ax4.plot(center_x, center_y, 'r+', markersize=15, markeredgewidth=3)
    
    plt.tight_layout()
    
    # Save the comprehensive analysis plot
    analysis_filepath = os.path.join(output_directory, "workpiece_projection_analysis.png")
    plt.savefig(analysis_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # Create a simple, clean workpiece projection plot
    fig2, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Calculate appropriate plot bounds with padding for better visibility
    x_range = max(x_vals) - min(x_vals)
    y_range = max(y_vals) - min(y_vals)
    padding_factor = 0.3  # 30% padding around the pattern
    
    x_padding = max(x_range * padding_factor, 0.2)  # Minimum 0.2 units padding
    y_padding = max(y_range * padding_factor, 0.2)
    
    x_min = min(x_vals) - x_padding
    x_max = max(x_vals) + x_padding
    y_min = min(y_vals) - y_padding
    y_max = max(y_vals) + y_padding
    
    # Clean 2D projection with professional styling - optimized for high-resolution data
    point_size = max(20, min(100, 5000 // len(x_vals)))  # Dynamic sizing based on data density
    scatter = ax.scatter(x_vals, y_vals, c=times, cmap='viridis', s=point_size, alpha=0.8, 
                        edgecolors='white', linewidth=0.8)
    
    # Connect points to show scan trajectory - adjusted for data density
    line_width = max(0.5, min(2.0, 200 / len(x_vals)))  # Thinner lines for dense data
    ax.plot(x_vals, y_vals, 'k-', alpha=0.4, linewidth=line_width)
    
    # Add 95% containment circle
    circle = Circle((center_x, center_y), effective_diameter/2, 
                   fill=False, color='red', linestyle='--', linewidth=3.0)
    ax.add_patch(circle)
    
    # Mark center and start/end points - larger markers
    ax.plot(center_x, center_y, 'r+', markersize=20, markeredgewidth=4, label='Center')
    ax.plot(x_vals[0], y_vals[0], 'go', markersize=12, label=f'Start (t={times[0]:.1f}s)')
    ax.plot(x_vals[-1], y_vals[-1], 'ro', markersize=12, label=f'End (t={times[-1]:.1f}s)')
    
    # Set tight bounds for better zoom on the pattern
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    ax.set_xlabel('X Position (units)', fontsize=14, weight='bold')
    ax.set_ylabel('Y Position (units)', fontsize=14, weight='bold')
    ax.set_title('Risley Prism Workpiece Laser Projection', fontsize=16, weight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(fontsize=12, loc='upper right')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Time (seconds)', fontsize=12, weight='bold')
    
    # Add comprehensive statistics annotation
    stats_text = f'Scan Statistics:\n'
    stats_text += f'• Area: {(max(x_vals)-min(x_vals))*(max(y_vals)-min(y_vals)):.2f} sq units\n'
    stats_text += f'• X Range: {max(x_vals)-min(x_vals):.2f} units\n'
    stats_text += f'• Y Range: {max(y_vals)-min(y_vals):.2f} units\n'
    stats_text += f'• Center: ({center_x:.2f}, {center_y:.2f})\n'
    stats_text += f'• Effective Diameter: {effective_diameter:.2f} units\n'
    stats_text += f'• Scan Points: {len(workpiece_projections)}\n'
    stats_text += f'• Duration: {max(times):.1f} seconds'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9),
            verticalalignment='top', fontsize=11, weight='bold')
    
    plt.tight_layout()
    
    # Save the clean workpiece projection plot
    projection_filepath = os.path.join(output_directory, "workpiece_projection.png")
    plt.savefig(projection_filepath, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Workpiece projection image saved to {projection_filepath}")
    print(f"Comprehensive analysis image saved to {analysis_filepath}")
