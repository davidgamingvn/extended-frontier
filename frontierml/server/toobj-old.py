# This script converts Apple's annoying USDZ format to obj, which is easier to work with
# It also only keeps the walls of the room, which the only part of the mesh that's useful for the WiFi coverage analysis
# Pip dependencies: usd-core, numpy

from pxr import Usd, UsdGeom, Gf
import numpy as np

def transform_points(points, matrix):
    """Transform points by a 4x4 matrix"""
    transformed_points = []
    for point in points:
        p = Gf.Vec3d(point[0], point[1], point[2])
        transformed = matrix.Transform(p)
        transformed_points.append(transformed)
    return transformed_points

def is_under_arch_group(prim_path):
    """Check if the prim is under the Arch_grp path"""
    path_str = str(prim_path)
    return path_str.startswith("/MergedRooms/Mesh_grp/Arch_grp")

def usdz_to_obj(usdz_file, obj_file):
    """
    Convert a USDZ file to OBJ format with correct transformations.
    Only processes meshes under /MergedRooms/Mesh_grp/Arch_grp

    Args:
        usdz_file (str): Path to input USDZ file
        obj_file (str): Path to output OBJ file
    """
    # Open the USD stage
    stage = Usd.Stage.Open(usdz_file)

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

        # Write header with information about the conversion
        f.write(f"# Converted from {usdz_file}\n")
        f.write(f"# Filtered to Arch_grp meshes only\n")
        f.write(f"# Original up axis: {up_axis}\n\n")

        processed_count = 0

        # Traverse all prims in the stage
        for prim in stage.Traverse():
            if prim.IsA(UsdGeom.Mesh) and is_under_arch_group(prim.GetPath()):
                processed_count += 1
                mesh = UsdGeom.Mesh(prim)

                # Get mesh attributes
                points = mesh.GetPointsAttr().Get()
                face_vertex_counts = mesh.GetFaceVertexCountsAttr().Get()
                face_vertex_indices = mesh.GetFaceVertexIndicesAttr().Get()

                if points is None or face_vertex_counts is None or face_vertex_indices is None:
                    continue

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
                normals = mesh.GetNormalsAttr().Get()
                if normals is not None:
                    # Transform normals (ignore translation part of matrix)
                    normal_transform = final_transform.ExtractRotationMatrix()
                    transformed_normals = transform_points(normals, Gf.Matrix4d().SetRotate(normal_transform))
                    for normal in transformed_normals:
                        f.write(f"vn {normal[0]} {normal[1]} {normal[2]}\n")

                # Write faces
                idx = 0
                has_normals = normals is not None
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

        return processed_count

if __name__ == "__main__":
    usdz_file = "MergedRooms.usdz"
    obj_file = "output_obj_file.obj"

    try:
        processed_count = usdz_to_obj(usdz_file, obj_file)
        print(f"Successfully converted {processed_count} meshes from Arch_grp to {obj_file}")
    except Exception as e:
        print(f"Error converting file: {str(e)}")
