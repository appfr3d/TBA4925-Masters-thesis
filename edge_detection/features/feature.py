import copy
import numpy as np
import open3d as o3d
from helpers import dist, mean_dist

class Feature():
  def __init__(self, cloud: o3d.geometry.PointCloud) -> None:
    self.cloud = copy.deepcopy(cloud)

NUM_SCALES = 8

class ScalableFeature(Feature):
  def run(self):
    # Scale:
    #   "large" scale as 10% of BB diagonal 
    #   "small" scale as mean distance of each points to it's 10 nearest neighbors

    # Use these to get large scale
    BB = o3d.geometry.OrientedBoundingBox.create_from_points(self.cloud.points)
    corner_points = BB.get_box_points()

    distances = np.zeros(28)
    i = 0
    for p_i in range(len(corner_points)):
      for p_j in range(p_i+1, len(corner_points)):
        distances[i] = dist(corner_points[p_i], corner_points[p_j])
        i += 1

    max_scale = np.max(distances) / 10 # 10% of BB diagonal
    kd_tree = o3d.geometry.KDTreeFlann(cloud)

    points = np.asarray(cloud.points)
    mean_distances = np.zeros(points.shape[0])
    for point_i, point in enumerate(points):
      [_, idx, _] = kd_tree.search_knn_vector_3d(point, 11) # 10 nearest neighbors, and itself
      mean_distances[point_i] = mean_dist(point, points[idx[1:]])
    
    min_scale = np.mean(mean_distances)

    step = (max_scale-min_scale)/NUM_SCALES
    scales = [min_scale + step*i for i in range(NUM_SCALES)]

    for scale in scales:
      self.run_at_sacale(scale)

  def run_at_sacale(self, scale=float):
    pass

class GlobalFeature(Feature):
  def run(self):
    pass



class VoxelFeature():
  def __init__(self, cloud: o3d.geometry.PointCloud) -> None:
    self.cloud = copy.deepcopy(cloud)


  def run(self):
    # Scale:
    #   "large" scale as 10% of BB diagonal 
    #   "small" scale as mean distance of each points to it's 10 nearest neighbors

    # Use these to get large scale
    BB = o3d.geometry.OrientedBoundingBox.create_from_points(self.cloud.points)
    corner_points = BB.get_box_points()

    distances = np.zeros(28)
    i = 0
    for p_i in range(len(corner_points)):
      for p_j in range(p_i+1, len(corner_points)):
        distances[i] = dist(corner_points[p_i], corner_points[p_j])
        i += 1

    max_scale = np.max(distances) / 10 # 10% of BB diagonal
    kd_tree = o3d.geometry.KDTreeFlann(cloud)

    points = np.asarray(cloud.points)
    mean_distances = np.zeros(points.shape[0])
    for point_i, point in enumerate(points):
      [_, idx, _] = kd_tree.search_knn_vector_3d(point, 11) # 10 nearest neighbors, and itself
      mean_distances[point_i] = mean_dist(point, points[idx[1:]])
    
    min_scale = np.mean(mean_distances)

    step = (max_scale-min_scale)/NUM_SCALES
    scales = [min_scale + step*i for i in range(NUM_SCALES)]

    for scale in scales:
      self.run_at_sacale(scale)

  def run_at_sacale(self, scale=float):
    self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.cloud, voxel_size=scale)



if __name__ == "__main__":
  from helpers import read_roof_cloud, write_roof_cloud_result
  
  file_name = "32-1-510-215-53-roof-2-shift.ply"
  cloud = read_roof_cloud(file_name)
  s = ScalableFeature(cloud)
  s.run()
