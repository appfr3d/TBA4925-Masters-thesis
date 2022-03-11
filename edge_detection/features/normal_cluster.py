import numpy as np
import open3d as o3d
from feature import ScalableFeature
class NormalCluster(ScalableFeature):
  def preprocess_whole_cloud(self):
    # Add normals to cloud
    self.cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=80))
    self.cloud.orient_normals_consistent_tangent_plane(30)

    # Create KDTree
    self.kd_tree = o3d.geometry.KDTreeFlann(self.cloud)

  def run_at_sacale(self, scale=float):
    points = np.asarray(self.cloud.points)
    normals = np.asarray(self.cloud.normals)
    labels = np.zeros(points.shape[0])

    # Run through every point
    for point_i, point in enumerate(points):
      # Downscale cloud with ball query
      [k, idx, _] = self.kd_tree.search_radius_vector_3d(point, scale)

      # Set the points as the unit normals
      current_cloud = o3d.geometry.PointCloud()
      current_cloud.points = o3d.utility.Vector3dVector(normals[idx])

      # Cluster normals, minimum 20% of points needed to create a cluster
      current_labels = np.array(current_cloud.cluster_dbscan(eps=0.1, min_points=np.floor_divide(k, 5)))
      if current_labels[0] >= 0:
        labels[point_i] = -1
      else:
        # Store noise from dbscan as positive values
        labels[point_i] = 1

    return labels

if __name__ == "__main__":
  import os
  from helpers import read_roof_cloud, get_project_folder, save_scaled_feature_image
  file_name_base = "32-1-510-215-53-test-2"
  file_name = file_name_base + ".ply"
  cloud = read_roof_cloud(file_name)

  o3d.visualization.draw_geometries([cloud])

  f = NormalCluster(cloud)

  project_folder = get_project_folder()
  image_folder = os.path.join(project_folder, 'edge_detection/results/feature/normal_cluster/images/' + file_name_base + '/')
  
  # Create folder if not exists
  if not os.path.exists(image_folder):
    os.makedirs(image_folder)

  all_labels = f.run()

  # Create window
  vis = o3d.visualization.Visualizer()
  vis.create_window(width=1000, height=1000)
  vis.add_geometry(cloud)

  for label_i in range(all_labels.shape[0]):
    labels = all_labels[label_i]
    save_scaled_feature_image(vis, cloud, labels, image_folder, str(label_i))

  labels = np.sum(all_labels, axis=0)

  save_scaled_feature_image(vis, cloud, labels, image_folder, "Combined")

  vis.destroy_window()

  '''
  file_out_name = "32-1-510-215-53-normal_cluster-shift.ply"
  write_roof_cloud_result(file_name, cloud)
  '''


