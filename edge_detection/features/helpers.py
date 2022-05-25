from cProfile import label
import os
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import open3d as o3d

def get_project_folder():
  current_folder = os.path.dirname(os.path.abspath(__file__))
  return os.path.dirname(os.path.dirname(current_folder))

def get_roof_folder():
  project_folder = get_project_folder()
  return os.path.join(project_folder, "data/point_clouds/classified_roofs")

def read_roof_cloud(file_name):
  project_folder = get_project_folder()
  data_folder = os.path.join(project_folder, "data/point_clouds")
  roof_path = os.path.join(data_folder, "classified_roofs", file_name)

  cloud = o3d.io.read_point_cloud(roof_path)
  return cloud

def normalize_cloud(cloud):
  points = np.asarray(cloud.points)
  center = cloud.get_center()
  points -= center
  all_dist = np.array([dist([0,0,0], x) for x in points])
  max_dist = np.max(all_dist)
  points /= max_dist

  cloud.points = o3d.utility.Vector3dVector(points)
  return cloud, max_dist

def remove_noise(cloud):
  # Average dist to 10NN
  kd_tree = o3d.geometry.KDTreeFlann(cloud)
  points = np.asarray(cloud.points)
  mean_distances = np.zeros(points.shape[0])
  for point_i, point in enumerate(points):
    [_, idx, _] = kd_tree.search_knn_vector_3d(point, 11) # 10 nearest neighbors, and itself
    mean_d = mean_dist(point, points[idx[1:]])
    mean_distances[point_i] = mean_d

  min_scale = np.mean(mean_distances)
  labels = np.array(cloud.cluster_dbscan(eps=min_scale*2, min_points=12))
  clusters = np.unique(labels)
  
  # Visualize dbscan
  # color_labels = np.divide(labels + 1, clusters.shape[0] - 1)
  # test_cloud = o3d.geometry.PointCloud()
  # test_cloud.points = o3d.utility.Vector3dVector(points)
  # colormap = cm.get_cmap('rainbow') 
  # test_colors = np.array([colormap(l) for l in color_labels])
  # test_cloud.colors = o3d.utility.Vector3dVector(test_colors[:, :3])
  # o3d.visualization.draw_geometries([test_cloud], width=1024, height=1024)

  # Remove noisy clusters such as chimneys
  # E.i. every cluster with less than 100 points
  for cluster in clusters:
    if cluster >= 0:
      points_in_cluster = labels[labels == cluster]
      if points_in_cluster.shape[0] < 100:
        labels[labels == cluster] = -1

  # Visualize small cluster removal
  # roof_points = points[labels >= 0]
  # noise_points = points[labels < 0]
  # roof_cloud = o3d.geometry.PointCloud()
  # roof_cloud.points = o3d.utility.Vector3dVector(roof_points)
  # roof_cloud.colors = o3d.utility.Vector3dVector(np.zeros((roof_points.shape[0], 3)) + [0, 1, 0])
  # noise_cloud = o3d.geometry.PointCloud()
  # noise_cloud.points = o3d.utility.Vector3dVector(noise_points)
  # noise_cloud.colors = o3d.utility.Vector3dVector(np.zeros((noise_points.shape[0], 3)) + [1, 0, 0])
  # o3d.visualization.draw_geometries([roof_cloud, noise_cloud], width=1024, height=1024)

  # Final cloud
  final_cloud = o3d.geometry.PointCloud()
  final_cloud.points = o3d.utility.Vector3dVector(points[labels >= 0])
  final_cloud.colors = o3d.utility.Vector3dVector(np.asarray(cloud.colors)[labels >= 0])
  if cloud.has_normals():
    final_cloud.normals = o3d.utility.Vector3dVector(np.asarray(cloud.normals)[labels >= 0])

  return final_cloud


def write_roof_cloud_result(file_name, cloud):
  project_folder = get_project_folder()
  results_folder = os.path.join(project_folder, "data/point_clouds/edge_results")
  return o3d.io.write_point_cloud(os.path.join(results_folder, file_name), cloud, print_progress=True)

def save_scaled_feature_image(vis, cloud, labels, image_folder, scale_name):
  colormap = cm.get_cmap('rainbow') 
  colors = np.array([colormap(l) for l in labels])
  cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])

  vis.update_geometry(cloud)
  vis.update_renderer()
  vis.poll_events()

  if not os.path.exists(image_folder):
    os.makedirs(image_folder)

  img_path = os.path.join(image_folder, 'Scale' + scale_name + '.png')
  vis.capture_screen_image(img_path)


def dist(p1, p2):
  return np.sqrt(np.abs(p1[0] - p2[0])**2 + np.abs(p1[1] - p2[1])**2 + np.abs(p1[2] - p2[2])**2)

def mean_dist(p, pts):
  all_dist = np.array([dist(p, x) for x in pts])
  return np.mean(all_dist)

def max_dist(p, pts):
  all_dist = np.array([dist(p, x) for x in pts])
  return np.max(all_dist)

def max_mean_dist(p, pts):
  all_dist = np.array([dist(p, x) for x in pts])
  return np.max(all_dist), np.mean(all_dist)

def plane_intersect(a, b):
  """
  From: https://stackoverflow.com/questions/48126838/plane-plane-intersection-in-python
  a, b   4-tuples/lists
          Ax + By +Cz + D = 0
          A,B,C,D in order  

  Output: 2 points on line of intersection, np.arrays, shape (3,)
  """
  a_vec, b_vec = np.array(a[:3]), np.array(b[:3])

  aXb_vec = np.cross(a_vec, b_vec)

  A = np.array([a_vec, b_vec, aXb_vec])
  d = np.array([-a[3], -b[3], 0.]).reshape(3,1)

  # could add np.linalg.det(A) == 0 test to prevent linalg.solve throwing error

  p_inter = np.linalg.solve(A, d).T

  return p_inter[0], (p_inter + aXb_vec)[0]


def point_to_line_distance(P1, P2, X):
  """
  Math from: https://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html
  P1, P2: Two 3D points [x, y, z] that make up the line
  X: 3D point [x, y, z] to compute distance to line

  Output: distance, float
  """
  P1_vec = np.array(P1[:3])
  P2_vec = np.array(P2[:3])
  X_vec = np.array(X[:3])

  X_minus_P1 = X_vec - P1_vec
  X_minus_P2 = X_vec - P2_vec

  P2_minus_P1 = P2_vec - P1_vec

  numerator = np.linalg.norm(np.cross(X_minus_P1, X_minus_P2))
  denominator = np.linalg.norm(P2_minus_P1)

  return numerator/denominator
