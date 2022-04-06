import numpy as np
import open3d as o3d
from feature import Feature
from helpers import dist, mean_dist


#####
#       OBS: must normalize point clouds before saving them to use this feature!
#####

# TODO: change this to a scaled feature

class kNNCentroidDistance(Feature):
  def run(self):

    points = np.asarray(cloud.points)
    values = np.zeros(points.shape[0])

    # TODO: find good l and num_neighbors value, or use other technique to find them automatically
    l = 1
    num_neighbors = 70

    kd_tree = o3d.geometry.KDTreeFlann(cloud)

    for point_i, point in enumerate(points):
      [k, idx, _] = kd_tree.search_knn_vector_3d(point, num_neighbors)

      nearest_neighbors = points[idx[1:]]

      centroid = (1/(nearest_neighbors.shape[0])) * np.sum(nearest_neighbors, axis=0)

      # Get mean distance to 10 closest neighbors, then:
      min_dist = mean_dist(point, nearest_neighbors[:10])

      values[point_i] = dist(centroid, point) - l*min_dist

    return values
    

if __name__ == "__main__":
  import os
  from helpers import read_roof_cloud, get_project_folder, save_scaled_feature_image
  
  file_name_base = "32-1-510-215-53-test-1"
  file_name = file_name_base + ".ply"
  cloud = read_roof_cloud(file_name)

  f = kNNCentroidDistance(cloud)

  values = f.run()

  project_folder = get_project_folder()
  image_folder = os.path.join(project_folder, 'edge_detection/results/feature/knn_centroid_distance/images/' + file_name_base + '/')
  
  # Create folder if not exists
  if not os.path.exists(image_folder):
    os.makedirs(image_folder)

  # Create window
  vis = o3d.visualization.Visualizer()
  vis.create_window(width=1000, height=1000)
  vis.add_geometry(cloud)

  save_scaled_feature_image(vis, cloud, values, image_folder, "Global")

  # colors = np.zeros((values.shape[0], 3))
  # # colors += [0.6, 0.6, 0.6]
  # colors[values<=0] = [0.6, 0.6, 0.6]
  # colors[values>0] = [0, 1, 0] # Color points with high mean distance to knn centroid as green

  # cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

  # o3d.visualization.draw_geometries([cloud])

  '''
  file_out_name = "32-1-510-215-53-knn_centroid_distance-shift.ply"
  write_roof_cloud_result(file_out_name, cloud)
  '''


