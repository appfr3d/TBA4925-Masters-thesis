import copy
import numpy as np
import open3d as o3d
from helpers import dist, mean_dist

class Feature():
  def __init__(self, cloud: o3d.geometry.PointCloud) -> None:
    self.cloud = copy.deepcopy(cloud)
    self.preprocess_whole_cloud()

  def preprocess_whole_cloud(self):
    pass

NUM_SCALES = 8
VISUALIZE_VOXELS = False

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
    kd_tree = o3d.geometry.KDTreeFlann(self.cloud)

    points = np.asarray(self.cloud.points)
    mean_distances = np.zeros(points.shape[0])
    for point_i, point in enumerate(points):
      [_, idx, _] = kd_tree.search_knn_vector_3d(point, 11) # 10 nearest neighbors, and itself
      mean_distances[point_i] = mean_dist(point, points[idx[1:]])
    
    min_scale = np.mean(mean_distances)

    step = (max_scale-min_scale)/NUM_SCALES
    scales = [min_scale + step*i for i in range(NUM_SCALES)]

    # Preprocess if needed
    self.preprocess_whole_cloud()

    labels = np.zeros((NUM_SCALES, points.shape[0]))
    for scale_i in range(len(scales)):
      print('Calculating scale', scale_i)
      scale_labels = self.run_at_scale(scales[scale_i])
      labels[scale_i] = scale_labels
    
    return labels

  def preprocess_whole_cloud(self):
    pass

  def run_at_scale(self, scale=float):
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
    self.kd_tree = o3d.geometry.KDTreeFlann(self.cloud)

    self.points = np.asarray(self.cloud.points)
    mean_distances = np.zeros(self.points.shape[0])
    for point_i, point in enumerate(self.points):
      [_, idx, _] = self.kd_tree.search_knn_vector_3d(point, 11) # 10 nearest neighbors, and itself
      mean_distances[point_i] = mean_dist(point, self.points[idx[1:]])
    
    min_scale = np.mean(mean_distances)

    step = (max_scale-min_scale)/NUM_SCALES
    scales = [min_scale + step*i for i in range(NUM_SCALES)]

    # Preprocess if needed
    self.preprocess_whole_cloud()

    labels = np.zeros((NUM_SCALES, self.points.shape[0]))
    for scale_i in range(len(scales)):
      print('Calculating scale', scale_i, 'with size:', scales[scale_i])
      # Generate voxels
      self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.cloud, voxel_size=scales[scale_i])

      # Hash which points are in each grid index
      self.grid_index_to_point_indices = {}
      for point_i in range(self.points.shape[0]):
        grid_index = tuple(self.voxel_grid.get_voxel(self.points[point_i]))
        if not grid_index in self.grid_index_to_point_indices.keys():
          self.grid_index_to_point_indices[grid_index] = [point_i]
        else:
          self.grid_index_to_point_indices[grid_index].append(point_i)

      # Run feature
      scale_labels = self.run_at_scale(scales[scale_i], visualize=VISUALIZE_VOXELS)
      labels[scale_i] = scale_labels
    
    return labels

  def preprocess_whole_cloud(self):
    pass

  def run_at_scale(self, scale=float, visualize=True):
    pass
    

class SmallVoxelFeature():
  def __init__(self, cloud: o3d.geometry.PointCloud) -> None:
    self.cloud = copy.deepcopy(cloud)
  
  def run(self):
    self.kd_tree = o3d.geometry.KDTreeFlann(self.cloud)

    self.points = np.asarray(self.cloud.points)
    mean_distances = np.zeros(self.points.shape[0])
    for point_i, point in enumerate(self.points):
      [_, idx, _] = self.kd_tree.search_knn_vector_3d(point, 11) # 10 nearest neighbors, and itself
      mean_distances[point_i] = mean_dist(point, self.points[idx[1:]])
    
    min_scale = np.mean(mean_distances)
    scales = [min_scale]

    # Preprocess if needed
    self.preprocess_whole_cloud()

    labels = np.zeros((len(scales), self.points.shape[0]))
    for scale_i in range(len(scales)):
      print('Calculating scale', scale_i, 'with size:', scales[scale_i])
      # Generate voxels
      self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.cloud, voxel_size=scales[scale_i])

      # Hash which points are in each grid index
      self.grid_index_to_point_indices = {}
      for point_i in range(self.points.shape[0]):
        grid_index = tuple(self.voxel_grid.get_voxel(self.points[point_i]))
        if not grid_index in self.grid_index_to_point_indices.keys():
          self.grid_index_to_point_indices[grid_index] = [point_i]
        else:
          self.grid_index_to_point_indices[grid_index].append(point_i)

      # Run feature
      scale_labels = self.run_at_scale(scales[scale_i], visualize=VISUALIZE_VOXELS)
      labels[scale_i] = scale_labels
    
    return labels

  def preprocess_whole_cloud(self):
    pass

  def run_at_scale(self, scale=float, visualize=True):
    pass


if __name__ == "__main__":
  from helpers import read_roof_cloud
  
  file_name = "32-1-510-215-53-roof-2-shift.ply"
  cloud = read_roof_cloud(file_name)
  s = VoxelFeature(cloud)

  # Visualize point cloud first
  o3d.visualization.draw([s.cloud])
  s.run()
