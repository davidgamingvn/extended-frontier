from flask import Flask, request, render_template, jsonify, send_file
import os
import numpy as np
import cv2
from scipy.spatial import cKDTree
from toobj import usdz_to_obj
import time

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Create uploads folder in the same directory as the script
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
IMAGE_SIZE = 100

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def save_floor_plan(layout, interior):
    """Save the floor plan visualization as an image"""
    # Create a 3-channel image (white background)
    floor_plan = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 255

    # Draw walls in black
    floor_plan[layout > 0] = [0, 0, 0]

    # Draw interior space in light gray
    floor_plan[interior > 0] = [240, 240, 240]

    # Save the floor plan
    floor_plan_path = os.path.join(UPLOAD_FOLDER, 'latest_floor_plan.png')
    cv2.imwrite(floor_plan_path, floor_plan)
    return floor_plan_path

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

    for pos in device_positions:
        coverage[pos[0], pos[1]] = 1.0

    kernel_size = 5
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

        wall_factor = 0.5 + (0.3 * i / iterations)  # Starts at 0.5, gradually increases to 0.8
        new_coverage[layout > 0] *= wall_factor

        new_coverage[interior == 0] = 0

        new_coverage *= decay_factor

        weak_signals = (new_coverage > 0.1) & (new_coverage < 0.3)
        new_coverage[weak_signals] *= 1.1

        for pos in device_positions:
            new_coverage[pos[0], pos[1]] = 1.0

            y, x = pos
            radius = 2
            y_min, y_max = max(0, y-radius), min(coverage.shape[0], y+radius+1)
            x_min, x_max = max(0, x-radius), min(coverage.shape[1], x+radius+1)
            new_coverage[y_min:y_max, x_min:x_max] = np.maximum(
                new_coverage[y_min:y_max, x_min:x_max],
                0.8
            )

        coverage = new_coverage

    coverage = np.clip(coverage * 1.2, 0, 1)  # Boost signals slightly

    coverage = cv2.GaussianBlur(coverage, (5,5), 1)

    return coverage

def find_extender_positions(layout, interior, router_pos, n_extenders):
    """Find optimal positions for multiple extenders with updated coverage threshold"""
    extender_positions = []
    coverage_maps = []

    router_coverage = simulate_wifi(layout, [router_pos], interior)

    for _ in range(n_extenders):
        best_coverage = 0
        best_pos = None
        best_coverage_map = None

        y_coords, x_coords = np.where(interior > 0)

        for i in range(0, len(y_coords), 2):
            y, x = y_coords[i], x_coords[i]

            dist = np.sqrt((y - router_pos[0])**2 + (x - router_pos[1])**2)

            too_close = False
            if dist < IMAGE_SIZE * 0.25:  # Slightly reduced minimum distance
                continue

            for ext_pos in extender_positions:
                if np.sqrt((y - ext_pos[0])**2 + (x - ext_pos[1])**2) < IMAGE_SIZE * 0.25:
                    too_close = True
                    break

            if too_close:
                continue

            current_devices = [router_pos] + extender_positions + [(y, x)]
            total_coverage = simulate_wifi(layout, current_devices, interior)

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

    for i in range(0, len(y_coords), 2):
        y, x = y_coords[i], x_coords[i]

        dist = np.sqrt((y - router_pos[0])**2 + (x - router_pos[1])**2)

        if dist > IMAGE_SIZE * 0.3:
            total_coverage = simulate_wifi(layout, [router_pos, (y, x)], interior)

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
        # Ensure upload directory exists again just in case
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        file = request.files['file']
        router_x = float(request.form['router_x'])
        router_y = float(request.form['router_y'])
        n_extenders = int(request.form.get('n_extenders', 2))

        usdz_path = os.path.join(UPLOAD_FOLDER, 'input.usdz')
        obj_path = os.path.join(UPLOAD_FOLDER, 'converted.obj')

        print(f"Saving files to: {UPLOAD_FOLDER}")
        print(f"USDZ path: {usdz_path}")
        print(f"OBJ path: {obj_path}")

        file.save(usdz_path)

        processed_count = usdz_to_obj(usdz_path, obj_path)

        layout, interior = convert_obj_to_2d(obj_path)

        # Save floor plan after conversion
        floor_plan_path = save_floor_plan(layout, interior)

        router_pos = (
            int(router_y * (IMAGE_SIZE-1)),
            int(router_x * (IMAGE_SIZE-1))
        )

        router_coverage = simulate_wifi(layout, [router_pos], interior)

        extender_positions, coverage_maps = find_extender_positions(
            layout, interior, router_pos, n_extenders
        )

        if coverage_maps:
            combined_coverage = coverage_maps[-1]
        else:
            combined_coverage = router_coverage

        coverage_score = float(np.sum(combined_coverage > 0.2)) / (IMAGE_SIZE * IMAGE_SIZE)

        heatmap = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        coverage_vis = (combined_coverage * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(coverage_vis, cv2.COLORMAP_TWILIGHT)

        heatmap[interior == 0] = [0, 0, 0]    # Black outside
        heatmap[layout > 0] = [255, 255, 255] # White walls

        cv2.circle(heatmap, (router_pos[1], router_pos[0]), 2, (255, 255, 255), -1)
        for pos in extender_positions:
            cv2.circle(heatmap, (pos[1], pos[0]), 2, (0, 255, 0), -1)

        coverage_image_path = os.path.join(UPLOAD_FOLDER, 'latest_coverage.png')
        print(f"Saving coverage image to: {coverage_image_path}")
        success = cv2.imwrite(coverage_image_path, heatmap)

        if not success:
            print("Failed to save coverage image!")
            return jsonify({'error': 'Failed to save coverage image'}), 500

        if not os.path.exists(coverage_image_path):
            print("Coverage image file does not exist after saving!")
            return jsonify({'error': 'Coverage image not found after saving'}), 500

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
        print(f"Error in analyze route: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_image')
@app.route('/get_image/<timestamp>')
def get_image(timestamp=None):
    coverage_image_path = os.path.join(UPLOAD_FOLDER, 'latest_coverage.png')
    print(f"Attempting to serve image from: {coverage_image_path}")

    # Create a default image if none exists
    if not os.path.exists(coverage_image_path):
        print("Coverage image not found, creating default image")
        default_image = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(default_image, 'No analysis yet', (10, IMAGE_SIZE//2),
                   font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Ensure the uploads directory exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Save the default image
        success = cv2.imwrite(coverage_image_path, default_image)
        if not success:
            print("Failed to save default image!")
            return jsonify({'error': 'Failed to save default image'}), 500

    try:
        if not os.path.exists(coverage_image_path):
            print("Image still doesn't exist after attempted creation!")
            return jsonify({'error': 'Image file not found'}), 404

        response = send_file(
            coverage_image_path,
            mimetype='image/png'
        )
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        print(f"Error serving image: {str(e)}")
        return jsonify({'error': 'Unable to load coverage image'}), 500

@app.route('/get_floor_plan')
@app.route('/get_floor_plan/<timestamp>')
def get_floor_plan(timestamp=None):
    """Endpoint to fetch the latest floor plan image"""
    floor_plan_path = os.path.join(UPLOAD_FOLDER, 'latest_floor_plan.png')
    print(f"Attempting to serve floor plan from: {floor_plan_path}")

    # Create a default image if none exists
    if not os.path.exists(floor_plan_path):
        print("Floor plan not found, creating default image")
        default_image = np.ones((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8) * 255
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(default_image, 'No floor plan yet', (10, IMAGE_SIZE//2),
                   font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        # Ensure the uploads directory exists
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        # Save the default image
        success = cv2.imwrite(floor_plan_path, default_image)
        if not success:
            print("Failed to save default floor plan!")
            return jsonify({'error': 'Failed to save default floor plan'}), 500

    try:
        if not os.path.exists(floor_plan_path):
            print("Floor plan still doesn't exist after attempted creation!")
            return jsonify({'error': 'Floor plan file not found'}), 404

        response = send_file(
            floor_plan_path,
            mimetype='image/png'
        )
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        print(f"Error serving floor plan: {str(e)}")
        return jsonify({'error': 'Unable to load floor plan image'}), 500

if __name__ == '__main__':
    print(f"Server starting with upload folder at: {UPLOAD_FOLDER}")
    app.run(debug=True)
