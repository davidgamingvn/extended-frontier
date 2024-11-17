from pxr import Usd, UsdGeom
import numpy as np
from wifi_analyzer import WifiCoverageAnalyzer
import matplotlib.pyplot as plt

class USDZParser:
    def __init__(self, file_path):
        """
        Initialize USDZ parser
        Args:
            file_path: Path to the USDZ file
        """
        self.file_path = file_path
        self.stage = Usd.Stage.Open(file_path)
        if not self.stage:
            raise ValueError(f"Could not open USDZ file: {file_path}")

    def _is_wall_mesh(self, prim_path):
        """Check if the prim path corresponds to a wall mesh"""
        path_str = str(prim_path).lower()
        return 'wall' in path_str and 'arch' in path_str

    def parse_mesh(self):
        """
        Parse wall meshes from USDZ file
        Returns:
            tuple: (vertices, faces, normals, uvs)
        """
        # Get all wall meshes
        wall_meshes = []
        all_vertices = []
        all_faces = []
        vertex_offset = 0

        for prim in self.stage.TraverseAll():
            if prim.IsA(UsdGeom.Mesh) and self._is_wall_mesh(prim.GetPath()):
                wall_meshes.append(UsdGeom.Mesh(prim))
                print(f"Found wall mesh at: {prim.GetPath()}")

        if not wall_meshes:
            raise ValueError("No wall meshes found in the model")

        # Combine all wall meshes
        for mesh in wall_meshes:
            points = mesh.GetPointsAttr().Get()
            face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
            face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()

            if points is None or face_vertex_counts is None or face_vertex_indices is None:
                continue

            # Add vertices
            vertices = np.array([(p[0], p[1], p[2]) for p in points])
            all_vertices.append(vertices)

            # Add faces with offset
            faces = []
            idx = 0
            for count in face_vertex_counts:
                if count == 3:
                    faces.append([i + vertex_offset for i in face_vertex_indices[idx:idx+3]])
                elif count == 4:
                    # Split quad into two triangles
                    quad = [i + vertex_offset for i in face_vertex_indices[idx:idx+4]]
                    faces.append([quad[0], quad[1], quad[2]])
                    faces.append([quad[0], quad[2], quad[3]])
                idx += count
            all_faces.extend(faces)

            vertex_offset += len(vertices)

        # Combine all meshes
        vertices = np.vstack(all_vertices) if all_vertices else np.array([])
        faces = np.array(all_faces) if all_faces else np.array([])

        if len(vertices) == 0 or len(faces) == 0:
            raise ValueError("No valid wall mesh data found")

        # We don't need normals or UVs for wall analysis
        return vertices, faces, None, None

    def get_mesh_info(self):
        """
        Get information about wall meshes in the USDZ file
        Returns:
            str: Information about meshes
        """
        info = []
        wall_count = 0
        for prim in self.stage.TraverseAll():
            if prim.IsA(UsdGeom.Mesh):
                path = prim.GetPath()
                if self._is_wall_mesh(path):
                    wall_count += 1
                    mesh = UsdGeom.Mesh(prim)
                    points = mesh.GetPointsAttr().Get()
                    faces = mesh.GetFaceVertexCountsAttr().Get()
                    info.append(f"Wall mesh at path {path}:")
                    info.append(f"  Vertices: {len(points) if points else 0}")
                    info.append(f"  Faces: {len(faces) if faces else 0}")

        if wall_count == 0:
            info.append("No wall meshes found in the model")
        else:
            info.insert(0, f"Found {wall_count} wall meshes:")

        return "\n".join(info)

    def print_hierarchy(self):
        """
        Print the full hierarchy of the USDZ file
        """
        def _print_prim(prim, depth=0):
            print("  " * depth + str(prim.GetPath()))
            for child in prim.GetChildren():
                _print_prim(child, depth + 1)

        print("\nUSDZ Hierarchy:")
        _print_prim(self.stage.GetPseudoRoot())

def analyze_wifi_coverage(file_path, num_candidates, router_pos=(0.3, 0.3)):
    """
    Analyze WiFi coverage and find the best extender position
    Args:
        file_path: Path to the USDZ file
        num_candidates: Number of candidate positions to evaluate
        router_pos: Tuple of (x, y) coordinates for router position (normalized 0-1)
    """
    try:
        # Parse the 3D model
        print("Parsing 3D model...")
        parser = USDZParser(file_path)
        vertices, faces, _, _ = parser.parse_mesh()
        print(f"\nLoaded {len(vertices)} vertices and {len(faces)} faces from wall meshes")

        # Create WiFi coverage analyzer
        print("Creating coverage analyzer...")
        analyzer = WifiCoverageAnalyzer(vertices, faces, resolution=1025)

        # Create impedance map
        impedance_map = analyzer.create_impedance_map(wall_thickness=3)

        # Find best extender position
        print("Finding optimal extender position...")
        candidates = analyzer.find_best_extender_position(router_pos, num_candidates)
        if not candidates:
            print("No suitable positions found for extender")
            return

        # Get the best candidate
        best_candidate = candidates[0]
        best_pos = best_candidate['position']

        # Create figure
        fig = plt.figure(figsize=(15, 7))

        # Add a suptitle
        plt.suptitle('WiFi Coverage Analysis', fontsize=16, y=1.02)

        # Impedance map (left subplot)
        ax1 = plt.subplot(121)
        im1 = ax1.imshow(analyzer.impedance_map, cmap='gray')
        ax1.set_title('Floor Plan')
        plt.colorbar(im1, ax=ax1, label='Wall Impedance')

        # Coverage map (right subplot)
        ax2 = plt.subplot(122)

        # Get router and extender signals
        router_signal = analyzer.simulate_wifi_signal(router_pos)
        extender_signal = analyzer.simulate_wifi_signal(best_pos)

        # Combine signals
        combined_signal = np.maximum(router_signal, extender_signal)

        # Plot combined coverage
        im2 = ax2.imshow(combined_signal, cmap='jet')
        ax2.set_title('WiFi Signal Coverage')
        plt.colorbar(im2, ax=ax2, label='Signal Strength')

        # Plot router and extender positions
        router_x = int(router_pos[0] * analyzer.resolution)
        router_y = int(router_pos[1] * analyzer.resolution)
        ext_x = int(best_pos[0] * analyzer.resolution)
        ext_y = int(best_pos[1] * analyzer.resolution)

        # Plot with larger markers and better labels
        ax2.plot(router_x, router_y, 'w*', markersize=12, label='Router')
        ax2.plot(ext_x, ext_y, 'wo', markersize=12, label='Best Extender Position')

        # Add legend with better placement
        ax2.legend(loc='upper right', bbox_to_anchor=(1, -0.1))

        # Add coverage statistics as text
        coverage_stats = f"""
        Coverage Statistics:
        Router position: ({router_pos[0]:.2f}, {router_pos[1]:.2f})
        Best extender position: ({best_pos[0]:.2f}, {best_pos[1]:.2f})
        Coverage score: {best_candidate['score']:.0f}
        Coverage improvement: {(best_candidate['score'] / np.sum(router_signal > 0.3) - 1) * 100:.1f}%
        """

        plt.figtext(0.02, -0.1, coverage_stats, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

        # Adjust layout
        plt.tight_layout()

        # Show plot
        plt.show()

        # Print detailed analysis
        print("\nDetailed Analysis:")
        print(coverage_stats)

    except Exception as e:
        print(f"Error in analysis: {e}")
        raise
if __name__ == "__main__":
    file_path = "export.usdc"
    router_pos = (0.3, 0.3)  # You can adjust this

    analyze_wifi_coverage(file_path, 40, router_pos)
