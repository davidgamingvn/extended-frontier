from pxr import Usd, UsdGeom, Gf
import numpy as np

def transform_points(points, matrix):
    """Transform points by a 4x4 matrix"""
    if len(points) == 0:
        return []
    transformed_points = []
    for point in points:
        p = Gf.Vec3d(point[0], point[1], point[2])
        transformed = matrix.Transform(p)
        transformed_points.append(transformed)
    return transformed_points

def is_in_target_group(prim_path, target_paths):
    """
    Check if the prim is under any of the specified target paths

    Args:
        prim_path: Path to check
        target_paths: List of path prefixes to match against
    """
    path_str = str(prim_path)
    return any(path_str.startswith(prefix) for prefix in target_paths)

def usdz_to_obj(usdz_file, obj_file, target_paths=None, skip_empty=True):
    """
    Convert a USDZ file to OBJ format with correct transformations.

    Args:
        usdz_file (str): Path to input USDZ file
        obj_file (str): Path to output OBJ file
        target_paths (list): List of path prefixes to process. If None, processes all meshes
        skip_empty (bool): Skip meshes with no points/faces if True
    """
    # Open the USD stage
    stage = Usd.Stage.Open(usdz_file)
    if not stage:
        raise RuntimeError(f"Failed to open USD stage: {usdz_file}")

    # Create an XformCache for efficient computation of world transforms
    xform_cache = UsdGeom.XformCache()

    # Get stage up axis
    up_axis = UsdGeom.GetStageUpAxis(stage)

    # Define rotation matrix based on up axis
    if up_axis == UsdGeom.Tokens.y:
        rotation_matrix = Gf.Matrix4d().SetIdentity()
    elif up_axis == UsdGeom.Tokens.z:
        rotation_matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(1, 0, 0), -90))
    else:  # X-up
        rotation_matrix = Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), -90))

    # Prepare OBJ file for writing
    with open(obj_file, 'w') as f:
        vertex_offset = 1  # OBJ indices start at 1
        processed_count = 0
        skipped_count = 0

        # Write header with information about the conversion
        f.write(f"# Converted from {usdz_file}\n")
        if target_paths:
            f.write("# Filtered to paths:\n")
            for path in target_paths:
                f.write(f"#   {path}\n")
        f.write(f"# Original up axis: {up_axis}\n\n")

        # Traverse all prims in the stage
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Mesh):
                continue

            if target_paths and not is_in_target_group(prim.GetPath(), target_paths):
                continue

            mesh = UsdGeom.Mesh(prim)

            # Get mesh attributes
            try:
                points = mesh.GetPointsAttr().Get()
                face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
                face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()

                if points is None or face_vertex_counts is None or face_vertex_indices is None:
                    print(f"Warning: Skipping mesh with missing data: {prim.GetPath()}")
                    skipped_count += 1
                    continue

                if skip_empty and (len(points) == 0 or len(face_vertex_counts) == 0):
                    print(f"Warning: Skipping empty mesh: {prim.GetPath()}")
                    skipped_count += 1
                    continue

            except Exception as e:
                print(f"Error reading mesh data for {prim.GetPath()}: {str(e)}")
                skipped_count += 1
                continue

            processed_count += 1

            # Get world transform for this mesh
            world_transform = xform_cache.GetLocalToWorldTransform(prim)

            # Combine world transform with up-axis rotation
            final_transform = rotation_matrix * world_transform

            # Transform points to world space and apply up-axis rotation
            transformed_points = transform_points(points, final_transform)

            # Write vertex positions
            f.write(f"# Mesh: {prim.GetPath()}\n")
            for point in transformed_points:
                f.write(f"v {point[0]} {point[1]} {point[2]}\n")

            # Get normals if available
            try:
                normals = mesh.GetNormalsAttr().Get()
                if normals is not None and len(normals) > 0:
                    # Transform normals (ignore translation part of matrix)
                    normal_transform = final_transform.ExtractRotationMatrix()
                    transformed_normals = transform_points(normals, Gf.Matrix4d().SetRotate(normal_transform))
                    for normal in transformed_normals:
                        f.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")
            except Exception as e:
                print(f"Warning: Error processing normals for {prim.GetPath()}: {str(e)}")
                normals = None

            # Write faces
            idx = 0
            has_normals = normals is not None and len(normals) > 0
            for face_vertex_count in face_vertex_counts:
                f.write("f")
                for i in range(face_vertex_count):
                    vert_idx = face_vertex_indices[idx] + vertex_offset
                    if has_normals:
                        f.write(f" {vert_idx}//{vert_idx}")  # vertex//normal
                    else:
                        f.write(f" {vert_idx}")
                    idx += 1
                f.write("\n")

            # Update vertex offset for next mesh
            vertex_offset += len(points)

        return processed_count, skipped_count

if __name__ == "__main__":
    usdz_file = "Room.usdz"
    obj_file = "output_room.obj"

    # Define the paths you want to process
    target_paths = [
        "/Room/Parametric_grp/Arch_grp/Wall_grp"
    ]

    try:
        processed_count, skipped_count = usdz_to_obj(usdz_file, obj_file, target_paths=target_paths)
        print(f"Successfully converted {processed_count} meshes to {obj_file}")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} meshes due to errors or empty geometry")
    except Exception as e:
        print(f"Error converting file: {str(e)}")
