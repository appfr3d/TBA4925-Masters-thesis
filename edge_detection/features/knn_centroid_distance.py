import numpy as np
import open3d as o3d
from features.feature import ScalableFeature, ScalableFeatureState
from features.helpers import dist # , mean_dist
import copy


#####
#       OBS: must normalize point clouds before saving them to use this feature!
#####

class kNNCentroidDistance(ScalableFeature):
  def run_at_scale(self, scale:float, knn_scale:int):
    labels = np.zeros(self.state.points.shape[0])

    num_neighbors = knn_scale

    for point_i, point in enumerate(self.state.points):
      [k, idx, _] = self.state.kd_tree.search_knn_vector_3d(point, num_neighbors + 1)

      nearest_neighbors = self.state.points[idx[1:]]

      centroid = (1/(nearest_neighbors.shape[0])) * np.sum(nearest_neighbors, axis=0)

      # Visualize query point, neighborhood and centroid in different colors
      # q_pcd = o3d.geometry.PointCloud()
      # q_pcd.points = o3d.utility.Vector3dVector(np.array([centroid]))
      # q_pcd.paint_uniform_color([1,0,0]) # Centroid in red

      # c_pcd = o3d.geometry.PointCloud()
      # c_pcd.points = o3d.utility.Vector3dVector(self.state.points)

      # c_colors = copy.deepcopy(np.asarray(self.state.cloud.colors))
      # c_colors[idx[1:]] = [0, 1, 0] # Neighborhood in green
      # c_colors[point_i] = [0, 0, 1] # Query point in blue
      # c_pcd.colors = o3d.utility.Vector3dVector(c_colors)

      # o3d.visualization.draw_geometries([c_pcd, q_pcd], width=1024, height=1024)

      labels[point_i] = dist(centroid, point)


    labels *= self.state.downsampling_factor
    # Post process to correct lables
    # min_l = np.min(labels)
    # max_l = np.max(labels) - min_l

    # Other scaling options
    # (1/max_l)*eigen
    # k = 1
    # np.arctan(k*eigen)/np.pi*0.5
    # labels = (1/max_l)*(labels-min_l)

    '''
    min_l = np.min(labels)
    labels +=  np.abs(min_l)
    max_l = np.max(labels)

    scale_fn = lambda val: (1/max_l)*val
    labels_scaled = scale_fn(labels)
    '''

    return labels
    

if __name__ == "__main__":
  from helpers import read_roof_cloud, normalize_cloud
  file_name = "32-1-510-215-53-test-1.ply"
  cloud = read_roof_cloud(file_name)
  cloud = normalize_cloud(cloud)
  print(cloud)

  state = ScalableFeatureState(cloud)
  f = kNNCentroidDistance(state)
  f.run_test('knn_centroid_distance', file_name)


