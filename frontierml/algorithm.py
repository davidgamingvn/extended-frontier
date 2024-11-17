import numpy as np
from scipy.spatial import cKDTree
import random
import matplotlib.pyplot as plt

def calculate_adaptive_min_distance(image, n_points, padding_factor=0.8):
    """
    Calculate a suitable minimum distance based on image size and number of points.

    Args:
        image: Binary image array
        n_points: Target number of points
        padding_factor: Adjustment factor (lower means points can be closer)
    """
    available_area = np.sum(image == 0)
    area_per_point = available_area / n_points

    # Approximate each point's territory as a circle
    # min_distance is the diameter of this circle
    min_distance = np.sqrt(area_per_point / np.pi) * 2 * padding_factor

    return max(2, min_distance)  # Ensure minimum of 2 pixels

def find_and_visualize_points(image, n_points, min_distance=None, visualize=True):
    """
    Find evenly distributed points and visualize them.

    Args:
        image: 2D numpy array (0 = available, 255 = obstacle)
        n_points: Target number of points
        min_distance: Optional minimum distance (calculated if None)
        visualize: Whether to show the plot

    Returns:
        points: List of (x, y) coordinates
        actual_min_distance: The minimum distance actually used
    """
    height, width = image.shape

    # Calculate adaptive minimum distance if not provided
    if min_distance is None:
        min_distance = calculate_adaptive_min_distance(image, n_points)

    # Find available positions
    available_positions = np.where(image == 0)
    available_coords = list(zip(available_positions[1], available_positions[0]))

    if not available_coords:
        raise ValueError("No available positions found in image")

    points = []
    attempts = 0
    max_attempts = n_points * 20

    # Start with a random point
    points.append(random.choice(available_coords))
    tree = cKDTree(np.array(points))

    while len(points) < n_points and attempts < max_attempts:
        candidate = random.choice(available_coords)

        if len(points) > 0:
            distances, _ = tree.query([candidate])
            if distances[0] < min_distance:
                attempts += 1

                # If we're struggling to place points, gradually reduce minimum distance
                if attempts % (max_attempts // 4) == 0:
                    min_distance *= 0.9
                continue

        points.append(candidate)
        tree = cKDTree(np.array(points))
        attempts = 0

    points = np.array(points)

    if visualize:
        plt.figure(figsize=(10, 10))

        # Show the image in grayscale
        plt.imshow(image, cmap='gray')

        # Plot points with different colors based on order
        scatter = plt.scatter(points[:, 0], points[:, 1],
                            c=range(len(points)),
                            cmap='viridis',
                            s=50)

        # Add colorbar to show point placement order
        plt.colorbar(scatter, label='Point placement order')

        plt.title(f'Distributed {len(points)} points\nMinimum distance: {min_distance:.1f} pixels')
        plt.axis('equal')
        plt.show()

    return points, min_distance

def create_test_image(width=200, height=200):
    """Create a test image with various obstacles"""
    image = np.zeros((height, width), dtype=np.uint8)

    # Add some geometric obstacles
    # Central rectangle
    image[height//3:2*height//3, width//3:2*width//3] = 255

    # Some circles
    cy, cx = height//4, width//4
    y, x = np.ogrid[-cy:height-cy, -cx:width-cx]
    mask = x*x + y*y <= (min(width, height)//8)**2
    image[mask] = 255

    # Some random obstacles
    rng = np.random.default_rng(42)
    random_obstacles = rng.random((height, width)) > 0.8
    image[random_obstacles] = 255

    return image

# Example usage
if __name__ == "__main__":
    # Create test image
    image = create_test_image(200, 200)

    # Try different numbers of points
    for n_points in [50, 100, 200]:
        points, min_dist = find_and_visualize_points(image, n_points)
        print(f"Placed {len(points)} points with minimum distance of {min_dist:.1f} pixels")
