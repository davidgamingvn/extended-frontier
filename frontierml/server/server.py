from flask import Flask, request, jsonify, send_file, render_template_string
import numpy as np
from PIL import Image
import io
import tempfile
import os
from dataclasses import dataclass
from typing import Tuple, List, Dict
import math

app = Flask(__name__)

@dataclass
class Coverage:
    strength: float  # Signal strength from 0-1
    x: int
    y: int

class WifiSimulator:
    def __init__(self, width: int, height: int, wall_positions: List[Tuple[int, int]]):
        self.width = width
        self.height = height
        self.grid = np.zeros((height, width))

        # Set walls
        for x, y in wall_positions:
            self.grid[y, x] = 1.0

        self.WALL_ATTENUATION = 0.4
        self.DISTANCE_DECAY = 0.1
        self.COVERAGE_THRESHOLD = 0.2

        # Pre-calculate distance matrix for faster propagation
        self.y_indices, self.x_indices = np.indices((height, width))

    def simulate_signal(self, router_x: int, router_y: int, extender_x: int = None, extender_y: int = None) -> np.ndarray:
        # Calculate signal using vectorized operations
        signal = np.zeros_like(self.grid)

        # Router signal
        distances = np.sqrt((self.x_indices - router_x)**2 + (self.y_indices - router_y)**2)
        signal = np.maximum(1 - distances * self.DISTANCE_DECAY, 0)

        # Account for walls using shortest path
        wall_mask = self.grid > 0
        signal[wall_mask] *= (1 - self.WALL_ATTENUATION)

        # Add extender if present
        if extender_x is not None and extender_y is not None:
            distances = np.sqrt((self.x_indices - extender_x)**2 + (self.y_indices - extender_y)**2)
            extender_signal = np.maximum(1 - distances * self.DISTANCE_DECAY, 0)
            extender_signal[wall_mask] *= (1 - self.WALL_ATTENUATION)
            signal = np.maximum(signal, extender_signal)

        return signal

    def find_best_extender_position(self, router_x: int, router_y: int) -> Tuple[int, int, float]:
        # Calculate router coverage first
        base_signal = self.simulate_signal(router_x, router_y)
        base_coverage = self.calculate_coverage(base_signal)

        # Sample fewer points for faster calculation
        step = 4  # Check every 4th point
        best_x, best_y = 0, 0
        best_coverage = base_coverage

        for y in range(0, self.height, step):
            for x in range(0, self.width, step):
                # Skip walls and positions too close to router
                if self.grid[y, x] == 1.0 or abs(x - router_x) + abs(y - router_y) < 5:
                    continue

                signal = self.simulate_signal(router_x, router_y, x, y)
                coverage = self.calculate_coverage(signal)

                if coverage > best_coverage:
                    best_coverage = coverage
                    best_x, best_y = x, y

        return best_x, best_y, best_coverage - base_coverage
def convert_obj_to_2d(obj_file: str, resolution: int = 512) -> Tuple[np.ndarray, Dict[str, float]]:
    """Convert 3D OBJ to 2D top-down view"""
    # Read OBJ file
    vertices = []
    faces = []
    with open(obj_file, 'r') as f:
        for line in f:
            if line.startswith('v '):
                _, x, y, z = line.split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith('f '):
                face = [int(v.split('/')[0]) - 1 for v in line.split()[1:]]
                faces.append(face)

    vertices = np.array(vertices)

    # Find bounds
    min_coords = np.min(vertices, axis=0)
    max_coords = np.max(vertices, axis=0)

    # Create scaling factors
    scale = resolution / max(max_coords[0] - min_coords[0], max_coords[2] - min_coords[2])

    # Create 2D grid
    grid = np.zeros((resolution, resolution))

    # Project vertices to 2D
    for face in faces:
        face_vertices = vertices[face]
        # Project to top-down view (x-z plane)
        points = [(int((v[0] - min_coords[0]) * scale),
                  int((v[2] - min_coords[2]) * scale)) for v in face_vertices]

        # Draw face on grid
        for i in range(len(points)):
            p1 = points[i]
            p2 = points[(i + 1) % len(points)]
            # Draw line between points
            for x, y in _bresenham_line(p1[0], p1[1], p2[0], p2[1]):
                if 0 <= x < resolution and 0 <= y < resolution:
                    grid[y, x] = 1

    return grid, {
        'scale': scale,
        'min_x': min_coords[0],
        'min_z': min_coords[2],
    }

def _bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """Bresenham's line algorithm"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1

    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy

    points.append((x, y))
    return points


HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>WiFi Coverage Analyzer</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/fabric.js/5.3.1/fabric.min.js"></script>
</head>
<body>
    <h1>WiFi Coverage Analyzer</h1>

    <input type="file" id="usdzFile" accept=".usdz">
    <p>Click canvas to place router:</p>
    <canvas id="floorplan" width="800" height="600" style="border:1px solid black"></canvas>
    <p>Router position: <span id="routerPos">Not placed</span></p>

    <button id="analyze" disabled>Analyze</button>

    <div id="results" style="display:none">
        <p>Extender position: <span id="extenderPos"></span></p>
        <p>Coverage improvement: <span id="coverage"></span></p>
        <img id="coverageMap">
    </div>

    <script>
        const canvas = new fabric.Canvas('floorplan');
        let router = null;

        canvas.on('mouse:down', function(options) {
            const pointer = canvas.getPointer(options.e);
            if (router) canvas.remove(router);
            router = new fabric.Circle({
                radius: 8,
                fill: 'red',
                left: pointer.x - 8,
                top: pointer.y - 8,
                selectable: false
            });
            canvas.add(router);
            document.getElementById('routerPos').textContent =
                `x: ${pointer.x.toFixed(2)}, y: ${pointer.y.toFixed(2)}`;
            updateAnalyzeButton();
        });

        document.getElementById('usdzFile').addEventListener('change', updateAnalyzeButton);

        function updateAnalyzeButton() {
            document.getElementById('analyze').disabled =
                !(document.getElementById('usdzFile').files.length > 0 && router);
        }

        document.getElementById('analyze').addEventListener('click', async function() {
            const file = document.getElementById('usdzFile').files[0];
            if (!file || !router) return;

            const formData = new FormData();
            formData.append('usdz_file', file);
            const pointer = canvas.getPointer({ target: router });
            formData.append('router_x', pointer.x);
            formData.append('router_y', pointer.y);

            try {
                const response = await fetch('/analyze_coverage', {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) throw new Error('Analysis failed');

                const result = await response.json();
                document.getElementById('extenderPos').textContent =
                    `x: ${result.extender_position.x.toFixed(2)}, y: ${result.extender_position.y.toFixed(2)}`;
                document.getElementById('coverage').textContent =
                    `${result.coverage_improvement.toFixed(1)}%`;
                document.getElementById('coverageMap').src = result.image;
                document.getElementById('results').style.display = 'block';
            } catch (error) {
                alert('Analysis failed. Please try again.');
                console.error(error);
            }
        });
    </script>
</body>
</html>
'''
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze_coverage', methods=['POST'])
def analyze_coverage():
    if 'usdz_file' not in request.files:
        return jsonify({'error': 'No USDZ file provided'}), 400

    # Get canvas coordinates
    canvas_x = float(request.form.get('router_x', 0))
    canvas_y = float(request.form.get('router_y', 0))

    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as tmpdir:
        # Save USDZ file
        usdz_path = os.path.join(tmpdir, 'input.usdz')
        obj_path = os.path.join(tmpdir, 'converted.obj')
        request.files['usdz_file'].save(usdz_path)

        try:
            processed_count = usdz_to_obj(usdz_path, obj_path)
            if processed_count == 0:
                return jsonify({'error': 'No valid meshes found in USDZ file'}), 400
        except Exception as e:
            return jsonify({'error': f'USDZ conversion failed: {str(e)}'}), 400

        # Convert to 2D
        grid, scale_info = convert_obj_to_2d(obj_path)

        # Scale canvas coordinates to real-world coordinates
        canvas_scale = 800 / grid.shape[1]  # 800 is canvas width
        grid_x = int(canvas_x / canvas_scale)
        grid_y = int(canvas_y / canvas_scale)

        # Initialize simulator
        simulator = WifiSimulator(
            grid.shape[1],
            grid.shape[0],
            [(x, y) for y, x in zip(*np.where(grid > 0))]
        )

        # Find best extender position
        ext_x, ext_y, coverage_improvement = simulator.find_best_extender_position(
            grid_x, grid_y
        )

        # Generate coverage map
        coverage_map = simulator.simulate_signal(grid_x, grid_y, ext_x, ext_y)

        # Create visualization
        viz = np.zeros((*grid.shape, 3))
        viz[grid > 0] = [1, 1, 1]  # Walls in white
        viz[grid == 0] = coverage_map[grid == 0, None] * np.array([0, 0, 1])  # Signal strength in blue

        # Mark router and extender
        viz[grid_y-2:grid_y+2, grid_x-2:grid_x+2] = [1, 0, 0]  # Red
        viz[ext_y-2:ext_y+2, ext_x-2:ext_x+2] = [0, 1, 0]  # Green

        # Convert to image
        img = Image.fromarray((viz * 255).astype(np.uint8))
        img_io = io.BytesIO()
        img.save(img_io, 'PNG')
        img_io.seek(0)

        # Convert grid coordinates back to canvas coordinates for display
        canvas_ext_x = ext_x * canvas_scale
        canvas_ext_y = ext_y * canvas_scale

        # Return results
        return jsonify({
            'extender_position': {
                'x': canvas_ext_x,
                'y': canvas_ext_y
            },
            'coverage_improvement': coverage_improvement * 100,
            'image': f'/get_coverage_image?t={np.random.randint(10000)}'  # Cache busting
        }), 200

@app.route('/get_coverage_image')
def get_coverage_image():
    """Serve the last generated coverage image"""
    return send_file(
        img_io,
        mimetype='image/png',
        as_attachment=True,
        download_name='coverage_map.png'
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)

@app.route('/get_coverage_image')
def get_coverage_image():
    """Serve the last generated coverage image"""
    return send_file(
        img_io,
        mimetype='image/png',
        as_attachment=True,
        download_name='coverage_map.png'
    )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
