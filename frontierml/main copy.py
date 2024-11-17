# Save this as main.py
from wifi_analyzer import WifiCoverageAnalyzer
from usdz_parser import USDZParser
import matplotlib.pyplot as plt
import numpy as np

def analyze_wifi_coverage(file_path, router_pos=(0.3, 0.3), num_candidates_input=50):
    """
    Analyze WiFi coverage and find the best extender position

    Args:
        file_path: Path to the USDZ file
        router_pos: Tuple of (x, y) coordinates for router position (normalized 0-1)
    """
    try:
        # Parse the USDZ file
        print("Parsing 3D model...")
        parser = USDZParser(file_path)
        vertices, faces, _, _ = parser.parse_mesh()

        # Create WiFi coverage analyzer
        print("Creating coverage analyzer...")
        analyzer = WifiCoverageAnalyzer(vertices, faces, resolution=256)




        # Create impedance map
        impedance_map = analyzer.create_impedance_map(wall_thickness=3)

        # Find best extender position
        print("Finding optimal extender position...")
        candidates = analyzer.find_best_extender_position(router_pos, num_candidates=num_candidates_input)

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
    file_path = "APPLE.usdz"
    router_pos = (0.3, 0.3)  # You can adjust this

    analyze_wifi_coverage(file_path, router_pos, 1)
