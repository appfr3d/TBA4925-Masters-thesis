import numpy as np
import trimesh
from trimesh.exchange.binvox import voxelize_mesh
from trimesh import voxel as vox

# attach to logger so trimesh messages will be printed to console
trimesh.util.attach_to_log()

# mesh objects can be created from existing faces and vertex data
mesh_1 = trimesh.Trimesh(vertices=[[0, 0, 0], [0, 0, 1], [0, 1, 0]], faces=[[0, 1, 2]])

# load mesh as a pointcloud from file path
point_cloud = trimesh.load('/Users/alfredla/Documents/skole/master/TBA4925-Master/data/point_clouds/classified_roofs/32-1-510-215-53-test-4.ply')

# voxelize mesh
# voxels = vox.VoxelGrid(point_cloud.encode())
print(type(point_cloud))

# Get classification value
classification = point_cloud.metadata['ply_raw']['vertex']['data']['scalar_Classification']

# preview mesh in an opengl window if you installed pyglet and scipy with pip
# mesh_2.show()

mesh = trimesh.Trimesh(point_cloud.vertices)

voxel_grid = voxelize_mesh(mesh)

voxel_boxes = voxel_grid.as_boxes()
voxel_boxes.show()
