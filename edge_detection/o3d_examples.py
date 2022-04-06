from operator import inv, invert
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import sys

# from features.plane_intersection import PlaneIntersection

# Read data
def read():
  pcd = o3d.io.read_point_cloud("../data/point_clouds/classified_roofs/32-1-510-215-53-test-1.ply")
  # print(pcd)
  # print(pcd.colors)
  # print(np.asarray(pcd.colors))
  return pcd

def write(cloud):
  return o3d.io.write_point_cloud("../data/point_clouds/classified_roofs/with_normal/32-1-510-215-53-test-1.ply", cloud, print_progress=True)
  pass

# Create Open3d point cloud from numpy array
def create(array):
  pcd = o3d.geometry.PointCloud()
  pcd.points = o3d.utility.Vector3dVector(array)
  return pcd

# Visualize point cloud
def visualize(cloud):
  o3d.visualization.draw_geometries([cloud])


# Voxel downsampling
def voxel_down(cloud):
  down_pcd = cloud.voxel_down_sample(voxel_size=0.1)
  print(down_pcd)
  return down_pcd

# Vertex normal estimation
def vertex_normal_estimation(cloud):
  cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.2, max_nn=30))
  cloud.orient_normals_consistent_tangent_plane(30)

  # Now you can get normals from: cloud.normals
  # Get it as numpy array with np.asarray(cloud.normals)
  return cloud


def segment_plane(cloud):
  plane_model, inliers = cloud.segment_plane(distance_threshold=0.02, ransac_n=3, num_iterations=100)
  print('plane_model', plane_model)
  visualize(cloud.select_by_index(inliers))
  visualize(cloud.select_by_index(inliers, invert=True))

# Run function
if __name__ == "__main__":
  cloud = read()

  # segment_plane(cloud)

  # For variables in the cloud object:
  # help(cloud)

  # down = voxel_down(cloud)

  norm = vertex_normal_estimation(cloud) # .normalize_normals()
  visualize(norm)
  write(norm)

  # cloud.points = norm.normals
  # visualize(cloud)
