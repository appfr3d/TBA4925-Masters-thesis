import numpy as np
import os
import open3d as o3d

def read_roof_cloud(file_name):
  current_folder = os.path.dirname(os.path.abspath(__file__))
  project_folder = os.path.dirname(os.path.dirname(current_folder))
  data_folder = os.path.join(project_folder, "data/point_clouds")
  roof_path = os.path.join(data_folder, "roofs", file_name)

  cloud = o3d.io.read_point_cloud(roof_path)

  return cloud

def write_roof_cloud_result(file_name, cloud):
  current_folder = os.path.dirname(os.path.abspath(__file__))
  project_folder = os.path.dirname(os.path.dirname(current_folder))
  results_folder = os.path.join(project_folder, "data/point_clouds/edge_results")
  return o3d.io.write_point_cloud(os.path.join(results_folder, file_name), cloud, print_progress=True)

def dist(p1, p2):
  return np.sqrt(np.abs(p1[0] - p2[0])**2 + np.abs(p1[1] - p2[1])**2 + np.abs(p1[2] - p2[2])**2)

def mean_dist(p, pts):
  all_dist = np.array([dist(p, x) for x in pts])
  return np.mean(all_dist)


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
