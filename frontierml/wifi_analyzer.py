import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import measure
import matplotlib.pyplot as plt
from pathlib import Path


class WifiCoverageAnalyzer:
    def __init__(self, vertices, faces, resolution=256):
        self.vertices = vertices
        self.faces = faces
        self.resolution = resolution
        self.impedance_map = None
        self.signal_map = None

    def project_to_2d(self):
        """Project 3D model to 2D floor plan"""
        # Find the principal axes using PCA
        centered = self.vertices - np.mean(self.vertices, axis=0)
        _, eigvecs = np.linalg.eigh(centered.T @ centered)

        # Assume the up direction is the eigenvector with smallest variance
        up_direction = eigvecs[:,0]

        # Project vertices onto the floor plane
        floor_normal = up_direction
        d = -np.mean(centered @ floor_normal)

        # Create projection matrix
        P = np.eye(3) - np.outer(floor_normal, floor_normal)

        # Project vertices
        projected_vertices = (P @ centered.T).T

        # Scale to fit resolution
        min_coords = np.min(projected_vertices[:, :2], axis=0)
        max_coords = np.max(projected_vertices[:, :2], axis=0)
        scale = (self.resolution - 1) / np.max(max_coords - min_coords)

        normalized = (projected_vertices[:, :2] - min_coords) * scale

        return normalized

    def create_impedance_map(self, wall_thickness=3):
            """Create 2D impedance map from projected vertices"""
            projected = self.project_to_2d()
            # Initialize impedance map with HIGH impedance (was zero/low before)
            self.impedance_map = np.ones((self.resolution, self.resolution))  # High impedance for empty space
            # Draw LOW impedance walls (was high before)
            for face in self.faces:
                points = projected[face].astype(int)
                # Draw wall segments with LOW impedance
                for i in range(len(points)):
                    p1 = points[i]
                    p2 = points[(i + 1) % len(points)]
                    # Ensure points are within bounds
                    p1 = np.clip(p1, 0, self.resolution - 1)
                    p2 = np.clip(p2, 0, self.resolution - 1)
                    # Draw thick line for wall with LOW impedance
                    rr, cc = draw_thick_line(p1[0], p1[1], p2[0], p2[1], wall_thickness)
                    # Only draw valid points
                    valid = (rr >= 0) & (rr < self.resolution) & (cc >= 0) & (cc < self.resolution)
                    self.impedance_map[cc[valid], rr[valid]] = 0.0  # Low impedance for walls (will appear white)
            # Smooth the impedance map slightly
            self.impedance_map = gaussian_filter(self.impedance_map, sigma=0.5)
            return self.impedance_map

    def simulate_wifi_signal(self, router_pos, max_distance=100):
        """Simulate WiFi signal propagation with wall attenuation"""
        if self.impedance_map is None:
            self.create_impedance_map()

        # Initialize signal strength map
        self.signal_map = np.zeros((self.resolution, self.resolution))

        # Convert router position to pixel coordinates
        router_x = int(router_pos[0] * self.resolution)
        router_y = int(router_pos[1] * self.resolution)

        # Create distance map from router
        y, x = np.ogrid[:self.resolution, :self.resolution]
        distance_map = np.sqrt((x - router_x)**2 + (y - router_y)**2)

        # Basic signal strength calculation (inverse square law)
        signal_strength = 1 / (1 + (distance_map / max_distance)**2)

        # Create attenuation map based on walls
        attenuation_map = np.exp(-5 * self.impedance_map)  # Exponential attenuation through walls

        # Apply attenuation effects
        self.signal_map = signal_strength * attenuation_map

        return self.signal_map

    def find_best_extender_position(self, router_pos, num_candidates=10):
        """Find optimal position for WiFi extender"""
        if self.signal_map is None:
            self.simulate_wifi_signal(router_pos)

        # Find areas with weak but non-zero signal
        weak_signal = (self.signal_map > 0.1) & (self.signal_map < 0.4)

        # Only consider positions in free space (not in walls)
        valid_positions = weak_signal & (self.impedance_map < 0.5)

        if not np.any(valid_positions):
            return None

        # Find connected regions
        labels = measure.label(valid_positions)
        regions = measure.regionprops(labels)

        # Score each potential position
        candidates = []
        for region in regions:
            y, x = region.centroid
            extender_pos = (x / self.resolution, y / self.resolution)

            # Simulate coverage with extender at this position
            extender_signal = self.simulate_wifi_signal(extender_pos)

            # Combined coverage score
            combined_coverage = np.maximum(self.signal_map, extender_signal)

            # Calculate coverage score emphasizing indoor coverage
            indoor_mask = self.impedance_map < 0.5  # Areas inside the house
            coverage_score = np.sum(combined_coverage[indoor_mask] > 0.3)

            candidates.append({
                'position': extender_pos,
                'score': coverage_score,
                'region_size': region.area
            })

        # Sort candidates by score
        candidates.sort(key=lambda x: x['score'], reverse=True)

        return candidates[:num_candidates]

def visualize(self, router_pos=None, extender_pos=None):
    """Visualize impedance map and signal coverage"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # Plot impedance map
    im1 = ax1.imshow(self.impedance_map, cmap='gray')
    ax1.set_title('Floor Plan (Impedance Map)')
    plt.colorbar(im1, ax=ax1)

    # Plot signal coverage
    if router_pos is not None:
        router_signal = self.simulate_wifi_signal(router_pos)

        if extender_pos is not None:
            extender_signal = self.simulate_wifi_signal(extender_pos)
            combined_signal = np.maximum(router_signal, extender_signal)
        else:
            combined_signal = router_signal

        im2 = ax2.imshow(combined_signal, cmap='jet')
        ax2.set_title('WiFi Signal Coverage')
        plt.colorbar(im2, ax=ax2)

        # Plot router and extender positions
        router_x = int(router_pos[0] * self.resolution)
        router_y = int(router_pos[1] * self.resolution)
        ax2.plot(router_x, router_y, 'w*', markersize=10, label='Router')

        if extender_pos is not None:
            ext_x = int(extender_pos[0] * self.resolution)
            ext_y = int(extender_pos[1] * self.resolution)
            ax2.plot(ext_x, ext_y, 'wo', markersize=10, label='Extender')

        ax2.legend()

    plt.tight_layout()
    plt.show()

def draw_thick_line(x0, y0, x1, y1, thickness):
    """Draw a thick line using Bresenham's algorithm"""
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)

    if dx > dy:
        steps = dx
    else:
        steps = dy

    if steps == 0:
        return np.array([x0]), np.array([y0])

    x_inc = (x1 - x0) / steps
    y_inc = (y1 - y0) / steps

    x_coords = []
    y_coords = []

    for i in range(int(steps) + 1):
        x = x0 + i * x_inc
        y = y0 + i * y_inc

        # Add thickness
        for t in range(-thickness//2, thickness//2 + 1):
            x_coords.extend([int(x + t), int(x + t), int(x + t)])
            y_coords.extend([int(y - 1), int(y), int(y + 1)])

    return np.array(x_coords), np.array(y_coords)

# Example usage
if __name__ == "__main__":
    # Load your USDZ model and get vertices/faces
    parser = USDZParser("APPLE.usdz")
    vertices, faces, _, _ = parser.parse_mesh()

    # Create analyzer
    analyzer = WifiCoverageAnalyzer(vertices, faces, resolution=256)

    # Create impedance map
    impedance_map = analyzer.create_impedance_map()

    # Set router position (normalized coordinates 0-1)
    router_pos = (0.3, 0.3)  # Example position

    # Find best extender position
    candidates = analyzer.find_best_extender_position(router_pos)

    if candidates:
        best_pos = candidates[0]['position']
        print(f"Best extender position: {best_pos}")
        print(f"Coverage score: {candidates[0]['score']}")

        # Visualize results
        analyzer.visualize(router_pos, best_pos)
    else:
        print("No suitable position found for extender")
