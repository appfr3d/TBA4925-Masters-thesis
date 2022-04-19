import numpy as np
import open3d as o3d
from features.feature import ScalableFeature, ScalableFeatureState



#####
#       OBS: must normalize point clouds before saving them to use this feature!
#             or else we will loose a lot of precision...
#####

VISUALIZE = True

class NormalCluster(ScalableFeature):
  def config(self):
    self.thresholds = list(0 for _ in self.state.scales)
  
  def visualize_process(self):
    # Visualize for presentation
    o3d.visualization.draw_geometries([self.state.cloud])

    # Show normals from (0,0,0)
    c = o3d.geometry.PointCloud()
    c.points = self.state.cloud.normals
    c.normals = self.state.cloud.normals

    center = o3d.geometry.PointCloud()
    s = np.asarray(self.state.cloud.points).shape
    center.points = o3d.utility.Vector3dVector(np.zeros(s))
    center.normals = self.state.cloud.normals
    center.colors = o3d.utility.Vector3dVector(np.zeros(s) + [1, 0, 0])
    o3d.visualization.draw_geometries([c, center])

    labels = np.array(c.cluster_dbscan(eps=0.1, min_points=100))
    colors = np.zeros((labels.shape[0], 3))
    colors += [0.6, 0.6, 0.6]
    colors[labels < 0] = [0, 1, 0] # Color positive values as green
    c.colors = o3d.utility.Vector3dVector(colors[:, :3])
    c.points = self.state.cloud.points
    o3d.visualization.draw_geometries([c])

  def run_at_scale(self, scale=float):
    normals = np.asarray(self.state.cloud.normals)
    labels = np.zeros(self.state.points.shape[0]) # Default as a non-edge

    # Run through every point
    for point_i, point in enumerate(self.state.points):
      # Downscale cloud with ball query
      [k, idx, _] = self.state.kd_tree.search_radius_vector_3d(point, scale)

      # Set the points as the unit normals
      current_cloud = o3d.geometry.PointCloud()
      current_cloud.points = o3d.utility.Vector3dVector(normals[idx])

      # Cluster normals, minimum 20% of points needed to create a cluster
      current_labels = np.array(current_cloud.cluster_dbscan(eps=0.1, min_points=np.floor_divide(k, 5)))

      if current_labels[0] == -1:
        # Store noise from dbscan as edges
        labels[point_i] = 1
    

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