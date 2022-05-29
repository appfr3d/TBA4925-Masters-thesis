import numpy as np
import open3d as o3d
from matplotlib import cm
import copy
from features.feature import ScalableFeature, ScalableFeatureState

#####
#       OBS: must normalize point clouds before saving them to use this feature!
#             or else we will loose a lot of precision...
#####

VISUALIZE = True

class NormalCluster(ScalableFeature):
  def config(self):
    self.thresholds = list(0 for _ in self.state.scales)
  
  def visualize_process(self, cloud = None):
    # Visualize for presentation
    if cloud == None:
      cloud = self.state.cloud
    o3d.visualization.draw_geometries([cloud], width=1024, height=1024)

    # Show normals from (0,0,0)
    c = o3d.geometry.PointCloud()
    c.points = cloud.normals
    c.normals = cloud.normals
    center = o3d.geometry.PointCloud()
    s = np.asarray(cloud.points).shape
    center.points = o3d.utility.Vector3dVector(np.zeros(s))
    center.normals = cloud.normals
    center.colors = o3d.utility.Vector3dVector(np.zeros(s) + [1, 0, 0])
    o3d.visualization.draw_geometries([c, center], width=1024, height=1024)

    # Visualize dbscan
    labels = np.array(c.cluster_dbscan(eps=0.05, min_points=np.floor_divide(np.asarray(cloud.points).shape[0], 5)))
    clusters = np.unique(labels)
    print("clusters:", clusters)
    color_labels = np.divide(labels + 1, clusters.shape[0] - 1)
    test_cloud = o3d.geometry.PointCloud()
    test_cloud.points = c.points
    colormap = cm.get_cmap('rainbow')
    test_colors = np.array([colormap(l) for l in color_labels])
    test_cloud.colors = o3d.utility.Vector3dVector(test_colors[:, :3])
    o3d.visualization.draw_geometries([test_cloud], width=1024, height=1024)

    # Visualize result
    colors = np.zeros((labels.shape[0], 3))
    colors += [0.6, 0.6, 0.6]
    colors[labels < 0] = [0, 1, 0] # Color positive values as green
    c.colors = o3d.utility.Vector3dVector(colors[:, :3])
    c.points = cloud.points
    o3d.visualization.draw_geometries([c], width=1024, height=1024)

  def run_at_scale(self, scale=float, knn_scale=int):
    labels = np.zeros(self.state.points.shape[0]) # Default as a non-edge

    print(f'Running at scale {scale}')

    # Run through every point
    for point_i, point in enumerate(self.state.points):
      # Downscale cloud with ball query
      [k, idx, _] = self.state.kd_tree.search_radius_vector_3d(point, scale)

      # Set the points as the unit normals
      current_cloud = o3d.geometry.PointCloud()
      current_cloud.points = o3d.utility.Vector3dVector(self.state.normals[idx])

      # Cluster normals, minimum 20% of points needed to create a cluster
      current_labels = np.array(current_cloud.cluster_dbscan(eps=0.05, min_points=np.floor_divide(k, 5)))
      # print('current labels:', current_labels)

      if current_labels[0] == -1:
        # Store noise from dbscan as edges
        labels[point_i] = 1
      
      # if scale > 0.15 and point_i == 100:
        # Visualize whole cloud with the neighborhood
        # test_cloud = o3d.geometry.PointCloud()
        # test_cloud.points = o3d.utility.Vector3dVector(self.state.points[idx])
        # test_cloud.normals = o3d.utility.Vector3dVector(self.state.normals[idx])
        # self.visualize_process(test_cloud)
        #   t_colors = copy.deepcopy(np.asarray(self.state.cloud.colors))
        #   t_colors[idx] = [0, 1, 0]
        #   test_cloud.colors = o3d.utility.Vector3dVector(t_colors)
        #   o3d.visualization.draw_geometries([test_cloud], width=1024, height=1024)

        #   clusters = np.unique(labels)
        #   color_labels = np.divide(labels + 1, clusters.shape[0] - 1)
        #   colormap = cm.get_cmap('rainbow') 
        #   test_colors = np.array([colormap(l) for l in color_labels])
        #   current_cloud.colors = o3d.utility.Vector3dVector(test_colors[:, :3])
        #   o3d.visualization.draw_geometries([current_cloud], width=1024, height=1024)

    

    return labels

if __name__ == "__main__":
  from helpers import read_roof_cloud, normalize_cloud
  file_name = "32-1-510-215-53-test-1.ply"
  cloud = read_roof_cloud(file_name)
  cloud = normalize_cloud(cloud)
  print(cloud)

  state = ScalableFeatureState(cloud)
  f = NormalCluster(state)
  f.run_test('normal_cluster', file_name)