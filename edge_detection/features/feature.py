import os
import copy
from xmlrpc.client import boolean
import numpy as np
import open3d as o3d
from helpers import dist, mean_dist, read_roof_cloud, get_project_folder, save_scaled_feature_image, normalize_cloud


NUM_SCALES = 8
VISUALIZE_VOXELS = True
class FeatureState():
  def __init__(self, cloud: o3d.geometry.PointCloud) -> None:
    self.cloud = copy.deepcopy(cloud)
    self.kd_tree = o3d.geometry.KDTreeFlann(self.cloud)
    self.points = np.asarray(self.cloud.points)

    # Add normals to cloud
    self.cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=80))
    self.cloud.orient_normals_consistent_tangent_plane(30)

    self.preprocess_whole_cloud()

  def preprocess_whole_cloud(self):
    # Creates self.scales
    pass

class ScalableFeatureState(FeatureState):
  def preprocess_whole_cloud(self):
    # Scale: "large" scale as 10% of BB diagonal 
    #        "small" scale as mean distance of each points to it's 10 nearest neighbors

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

    mean_distances = np.zeros(self.points.shape[0])
    for point_i, point in enumerate(self.points):
      [_, idx, _] = self.kd_tree.search_knn_vector_3d(point, 11) # 10 nearest neighbors, and itself
      mean_distances[point_i] = mean_dist(point, self.points[idx[1:]])
    min_scale = np.mean(mean_distances)
    step = (max_scale-min_scale)/NUM_SCALES

    self.scales = [min_scale + step*i for i in range(NUM_SCALES)]

class SmallScalableFeatureState(FeatureState):
  def preprocess_whole_cloud(self):
    mean_distances = np.zeros(self.points.shape[0])
    for point_i, point in enumerate(self.points):
      [_, idx, _] = self.kd_tree.search_knn_vector_3d(point, 11) # 10 nearest neighbors, and itself
      mean_distances[point_i] = mean_dist(point, self.points[idx[1:]])
    min_scale = np.mean(mean_distances)
    
    self.scales = [min_scale * 0.5, min_scale, min_scale * 1.5]

class Feature():
  def __init__(self, state: FeatureState) -> None:
    self.state = state

  def run(self):
    # Returns a list of labels in different scales
    pass

  def run_test(self, feature_name: str, test_file_name: str):
    print('Start running...')
    all_labels = self.run()
    print('Done running!')

    print('Start saving...')
    project_folder = get_project_folder()
    results_feature_folder = 'edge_detection/results/feature'
    test_file_name_base = test_file_name.split('.')[0]
    image_folder = os.path.join(project_folder, results_feature_folder, feature_name + '/images/' + test_file_name_base + '/')
    
    # Create folder if not exists
    if not os.path.exists(image_folder):
      os.makedirs(image_folder)

    # Create window
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1000, height=1000)
    vis.add_geometry(self.state.cloud)

    if len(all_labels.shape) == 1:
      # Only global scale of labels
      save_scaled_feature_image(vis, self.state.cloud, all_labels, image_folder, "Global")
    else:
      # Several scales of labels
      for label_i in range(all_labels.shape[0]):
        labels = all_labels[label_i]
        save_scaled_feature_image(vis, self.state.cloud, labels, image_folder, str(label_i))

      # Combine scales as last labels
      labels = np.sum(all_labels, axis=0)
      save_scaled_feature_image(vis, self.state.cloud, labels, image_folder, "Combined")

    vis.destroy_window()
    print('Done saving!')

class ScalableFeature(Feature):
  def run(self):
    labels = np.zeros((NUM_SCALES, self.state.points.shape[0]))
    for scale_i, scale in enumerate(self.state.scales):
      print('Calculating scale', scale_i)
      scale_labels = self.run_at_scale(scale)
      labels[scale_i] = scale_labels
    return labels

  def run_at_scale(self, scale=float):
    pass

class VoxelFeature(Feature):
  def run(self):
    labels = np.zeros((NUM_SCALES, self.state.points.shape[0]))
    for scale_i, scale in enumerate(self.state.scales):
      print('Calculating scale', scale_i, 'with size:', scale)
      # Generate voxels
      self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.state.cloud, voxel_size=scale)

      # Would be to memory-intensive to hold all the voxels in all the scales in memory, so calculate them each time instead
      # Hash which points are in each grid index
      self.grid_index_to_point_indices = {}
      for point_i in range(self.state.points.shape[0]):
        grid_index = tuple(self.voxel_grid.get_voxel(self.state.points[point_i]))
        if not grid_index in self.grid_index_to_point_indices.keys():
          self.grid_index_to_point_indices[grid_index] = [point_i]
        else:
          self.grid_index_to_point_indices[grid_index].append(point_i)

      # Run feature
      scale_labels = self.run_at_scale(scale, visualize=VISUALIZE_VOXELS)
      labels[scale_i] = scale_labels
    
    return labels

  def run_at_scale(self, scale=float, visualize=bool):
    pass

class SmallVoxelFeature(Feature):
  def run(self):
    labels = np.zeros((len(self.state.scales), self.state.points.shape[0]))
    for scale_i, scale in enumerate(self.state.scales):
      print('Calculating scale', scale_i, 'with size:', scale)
      # Generate voxels
      self.voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(self.state.cloud, voxel_size=scale)

      # Would be to memory-intensive to hold all the voxels in all the scales in memory, so calculate them each time instead
      # Hash which points are in each grid index
      self.grid_index_to_point_indices = {}
      for point_i in range(self.state.points.shape[0]):
        grid_index = tuple(self.voxel_grid.get_voxel(self.state.points[point_i]))
        if not grid_index in self.grid_index_to_point_indices.keys():
          self.grid_index_to_point_indices[grid_index] = [point_i]
        else:
          self.grid_index_to_point_indices[grid_index].append(point_i)

      # Run feature
      scale_labels = self.run_at_scale(scale, visualize=VISUALIZE_VOXELS)
      labels[scale_i] = scale_labels
    
    return labels

  def run_at_scale(self, scale=float, visualize=bool):
    pass


if __name__ == "__main__":
  from helpers import read_roof_cloud
  
  file_name = "32-1-510-215-53-roof-2-shift.ply"
  cloud = read_roof_cloud(file_name)
  s = VoxelFeature(cloud)

  # Visualize point cloud first
  o3d.visualization.draw([s.cloud])
  s.run()
