from flask import Flask, request, render_template, jsonify, send_file
import os
import numpy as np
import cv2
from scipy.spatial import cKDTree
from toobj import usdz_to_obj
import time

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
IMAGE_SIZE = 100
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def get_interior_mask(layout):
    """Get interior mask using flood fill from exterior"""
    mask = np.ones_like(layout)
    mask[layout > 0] = 0
    exterior = mask.copy()
    cv2.floodFill(exterior, None, (0,0), 0)
    return (exterior > 0).astype(np.uint8)
def convert_obj_to_2d(obj_file):
    """Convert 3D OBJ to 2D top-down view with 5% padding"""
    vertices = []
    faces = []

    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                _, x, y, z = line.split()
                vertices.append([float(x), float(z)])
            elif line.startswith('f '):
                face = [int(v.split('/')[0]) - 1 for v in line.split()[1:]]
                faces.append(face)

    vertices = np.array(vertices)
    min_coords = vertices.min(axis=0)
    max_coords = vertices.max(axis=0)

    # Add 5% padding to the coordinate range
    coord_range = max_coords - min_coords
    padding = coord_range * 0.05
    min_coords -= padding
    max_coords += padding

    # Scale vertices to image space with padding
    vertices = ((vertices - min_coords) / (max_coords - min_coords) * (IMAGE_SIZE-2) + 1).astype(int)

    img = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    for face in faces:
        points = vertices[face]
        points = points.reshape((-1, 1, 2))
        img = cv2.polylines(img, [points], True, 255, thickness=1)

    interior = get_interior_mask(img)
    return img, interior

def simulate_wifi(layout, device_positions, interior, iterations=50, decay_factor=0.92):
    """
    Simulate WiFi propagation with more gradual signal distribution

    Parameters:
    - layout: 2D array of walls
    - device_positions: list of (y,x) coordinates for WiFi devices
    - interior: mask of interior space
    - iterations: number of propagation steps (increased for better spread)
    - decay_factor: signal decay per step (increased for better reach)
    """
    coverage = np.zeros_like(layout, dtype=float)

    # Initialize device positions with full strength
    for pos in device_positions:
        coverage[pos[0], pos[1]] = 1.0

    # Larger kernel for more gradual spread
    kernel_size = 10
    kernel = np.array([
        [0.05, 0.1, 0.15, 0.1, 0.05],
        [0.1, 0.2, 0.25, 0.2, 0.1],
        [0.15, 0.25, 0.0, 0.25, 0.15],
        [0.1, 0.2, 0.25, 0.2, 0.1],
        [0.05, 0.1, 0.15, 0.1, 0.05]
    ])

    # Multiple propagation passes with different characteristics
    for i in range(iterations):
        # Apply propagation kernel
        new_coverage = cv2.filter2D(coverage, -1, kernel)

        # Progressive wall attenuation (walls affect signal less in initial propagation)
        wall_factor = 0.5 + (0.3 * i / iterations)  # Starts at 0.5, gradually increases to 0.8
        new_coverage[layout > 0] *= wall_factor

        # Keep signal within interior
        new_coverage[interior == 0] = 0

        # Apply distance-based decay
        new_coverage *= decay_factor

        # Boost weak signals slightly to maintain better coverage
        weak_signals = (new_coverage > 0.1) & (new_coverage < 0.3)
        new_coverage[weak_signals] *= 1.1

        # Ensure device positions maintain full strength
        for pos in device_positions:
            new_coverage[pos[0], pos[1]] = 1.0

            # Add small area of strong signal around devices
            y, x = pos
            radius = 2
            y_min, y_max = max(0, y-radius), min(coverage.shape[0], y+radius+1)
            x_min, x_max = max(0, x-radius), min(coverage.shape[1], x+radius+1)
            new_coverage[y_min:y_max, x_min:x_max] = np.maximum(
                new_coverage[y_min:y_max, x_min:x_max],
                0.8
            )

        coverage = new_coverage

    # Final normalization to ensure good distribution
    coverage = np.clip(coverage * 1.2, 0, 1)  # Boost signals slightly

    # Smooth transition between strong and weak signals
    coverage = cv2.GaussianBlur(coverage, (5,5), 1)

    return coverage

def find_extender_positions(layout, interior, router_pos, n_extenders):
    """Find optimal positions for multiple extenders with updated coverage threshold"""
    extender_positions = []
    coverage_maps = []

    # Initial coverage from router
    router_coverage = simulate_wifi(layout, [router_pos], interior)

    for _ in range(n_extenders):
        best_coverage = 0
        best_pos = None
        best_coverage_map = None

        y_coords, x_coords = np.where(interior > 0)

        # Sample interior positions
        for i in range(0, len(y_coords), 2):
            y, x = y_coords[i], x_coords[i]

            # Calculate distance from router
            dist = np.sqrt((y - router_pos[0])**2 + (x - router_pos[1])**2)

            # Only consider positions far enough from router and other extenders
            too_close = False
            if dist < IMAGE_SIZE * 0.25:  # Slightly reduced minimum distance
                continue

            for ext_pos in extender_positions:
                if np.sqrt((y - ext_pos[0])**2 + (x - ext_pos[1])**2) < IMAGE_SIZE * 0.25:
                    too_close = True
                    break

            if too_close:
                continue

            # Calculate coverage with this extender
            current_devices = [router_pos] + extender_positions + [(y, x)]
            total_coverage = simulate_wifi(layout, current_devices, interior)

            # Score based on total coverage area (lowered threshold for better distribution)
            coverage_score = np.sum(total_coverage > 0.15) / (IMAGE_SIZE * IMAGE_SIZE)

            if coverage_score > best_coverage:
                best_coverage = coverage_score
                best_pos = (y, x)
                best_coverage_map = total_coverage

        if best_pos is None:
            break

        extender_positions.append(best_pos)
        coverage_maps.append(best_coverage_map)

    return extender_positions, coverage_maps

def find_best_extender_position(layout, interior, router_pos):
    """Find the best position for a WiFi extender"""
    best_coverage = 0
    best_pos = None
    best_total_coverage = None

    router_coverage = simulate_wifi(layout, [router_pos], interior)
    y_coords, x_coords = np.where(interior > 0)

    # Sample interior positions
    for i in range(0, len(y_coords), 2):
        y, x = y_coords[i], x_coords[i]

        # Calculate distance from router
        dist = np.sqrt((y - router_pos[0])**2 + (x - router_pos[1])**2)

        # Only consider positions far enough from router (at least 30% of image size)
        if dist > IMAGE_SIZE * 0.3:
            # Simulate coverage with both devices
            total_coverage = simulate_wifi(layout, [router_pos, (y, x)], interior)

            # Score based on total coverage area
            coverage_score = float(np.sum(combined_coverage > 0.2)) / (IMAGE_SIZE * IMAGE_SIZE)

            if coverage_score > best_coverage:
                best_coverage = coverage_score
                best_pos = (y, x)
                best_total_coverage = total_coverage

    return best_pos, best_coverage, best_total_coverage

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        file = request.files['file']
        router_x = float(request.form['router_x'])
        router_y = float(request.form['router_y'])
        n_extenders = int(request.form.get('n_extenders', 2))

        # Save and convert files
        usdz_path = os.path.join(UPLOAD_FOLDER, 'input.usdz')
        obj_path = os.path.join(UPLOAD_FOLDER, 'converted.obj')
        file.save(usdz_path)

        processed_count = usdz_to_obj(usdz_path, obj_path)
        layout, interior = convert_obj_to_2d(obj_path)

        # Convert router coordinates to image space
        router_pos = (
            int(router_y * (IMAGE_SIZE-1)),
            int(router_x * (IMAGE_SIZE-1))
        )

        # First simulate router coverage
        router_coverage = simulate_wifi(layout, [router_pos], interior)

        # Find extender positions
        extender_positions, coverage_maps = find_extender_positions(
            layout, interior, router_pos, n_extenders
        )

        # Use the last coverage map (includes all devices)
        if coverage_maps:
            combined_coverage = coverage_maps[-1]
        else:
            combined_coverage = router_coverage

        # Calculate coverage score
        coverage_score = float(np.sum(combined_coverage > 0.2)) / (IMAGE_SIZE * IMAGE_SIZE)

        # Create visualization
        heatmap = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        coverage_vis = (combined_coverage * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(coverage_vis, cv2.COLORMAP_JET)

        # Mark boundaries
        heatmap[interior == 0] = [0, 0, 0]    # Black outside
        heatmap[layout > 0] = [255, 255, 255] # White walls

        # Mark router and extenders
        cv2.circle(heatmap, (router_pos[1], router_pos[0]), 2, (255, 255, 255), -1)
        for pos in extender_positions:
            cv2.circle(heatmap, (pos[1], pos[0]), 2, (0, 255, 0), -1)

        # In the analyze route, after creating the heatmap:
        coverage_image_path = os.path.join(UPLOAD_FOLDER, 'latest_coverage.png')
        cv2.imwrite(coverage_image_path, heatmap)

        # Normalize positions to 0-1 range
        extender_positions_normalized = [
            {'x': pos[1] / IMAGE_SIZE, 'y': pos[0] / IMAGE_SIZE}
            for pos in extender_positions
        ]

        return jsonify({
            'image': '/get_image',
            'extender_positions': extender_positions_normalized,
            'coverage_score': coverage_score
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_image')
@app.route('/get_image/<timestamp>')
def get_image(timestamp=None):
    coverage_image_path = os.path.join(UPLOAD_FOLDER, 'latest_coverage.png')
    return send_file(coverage_image_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
