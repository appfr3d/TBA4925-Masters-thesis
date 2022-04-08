import numpy as np
import open3d as o3d
import trimesh
from trimesh.exchange.binvox import voxelize_mesh
from trimesh import voxel as vox

# attach to logger so trimesh messages will be printed to console
trimesh.util.attach_to_log()

# mesh objects can be created from existing faces and vertex data
mesh_1 = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], faces=[[0, 1, 2]])

# load mesh as a pointcloud from file path
roof_path = '/Users/alfredla/Documents/skole/master/TBA4925-Master/data/point_clouds/classified_roofs/32-1-510-215-53-test-1.ply'

tri_cloud = trimesh.load(roof_path)
o3d_cloud = o3d.io.read_point_cloud(roof_path)


tri_points = tri_cloud.vertices
o3d_points = np.asarray(o3d_cloud.points)

print('points:')
print(tri_points)
print(o3d_points)
print('points are equal:', np.array_equal(tri_points, o3d_points))

# voxelize mesh
# voxels = vox.VoxelGrid(point_cloud.encode())
# print(type(tri_cloud))

# Get classification value
classification = tri_cloud.metadata['ply_raw']['vertex']['data']['scalar_Classification']

print(classification)
print(type(classification))

# preview mesh in an opengl window if you installed pyglet and scipy with pip
# mesh_2.show()

# mesh = trimesh.Trimesh(point_cloud.vertices)

# voxel_grid = voxelize_mesh(tri_cloud.vertices)

# voxel_boxes = voxel_grid.as_boxes()
# voxel_boxes.show()
