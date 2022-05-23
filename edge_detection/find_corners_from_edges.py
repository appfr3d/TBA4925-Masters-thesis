import os
import copy
import numpy as np
import trimesh
import open3d as o3d
from skimage.measure import LineModelND, ransac
from features.helpers import read_roof_cloud, get_roof_folder, normalize_cloud, mean_dist

test_file_names = ["32-1-510-215-53-test-1.ply", "32-1-510-215-53-test-2.ply"]

def get_point_lables(file_name):
  roof_folder = get_roof_folder()
  tri_cloud = trimesh.load(os.path.join(roof_folder, file_name))
  return tri_cloud.metadata["ply_raw"]["vertex"]["data"]["scalar_Classification"]


VISUALIZE = False


def main():
  for file_name in test_file_names:
    print("Finding corners in file:", file_name)
    cloud = read_roof_cloud(file_name)
    cloud = normalize_cloud(cloud)
    points = np.asarray(cloud.points)
    

    labels = get_point_lables(file_name)

    edges = points[labels > 0]
    

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(edges)
    kd_tree = o3d.geometry.KDTreeFlann(pcd)

    if VISUALIZE:
      # pcd.colors = o3d.utility.Vector3dVector(colors)
      o3d.visualization.draw_geometries([pcd])
    
    # When 90% of the points are on edges, continue
    point_num_threshold = edges.shape[0] * 0.9

    # point is on line when close enough to me within the mean of the mean to distance to each points nearest neighbors 10
    mean_distances = np.zeros(edges.shape[0])
    for point_i, point in enumerate(edges):
      [_, idx, _] = kd_tree.search_knn_vector_3d(point, 11) # 10 nearest neighbors, and itself
      mean_distances[point_i] = mean_dist(point, edges[idx[1:]])
    line_threshold = np.mean(mean_distances)



    planes = []
    rest = copy.deepcopy(pcd)

    # TODO: calculate best distance threshold here...
    threshold = 0.03

    while len(rest.points) > 100:
      # Calculate best fitted plane using RANSAC
      ransac(rest, LineModelND, min_samples=3, residual_threshold=line_threshold, max_trials=1000)
      plane_model, inliers = rest.segment_plane(distance_threshold=threshold, ransac_n=3, num_iterations=100)
      # print('plane_model', plane_model)

      # Save best fitted plane
      planes.append(plane_model)

      # Remove points in plane from rest
      rest = rest.select_by_index(inliers, invert=True)


    found_edge_points = {}
    remaining_edges = edges.copy()
    interation = 0
    while len(found_edge_points.keys()) < point_num_threshold:
      percent = (len(found_edge_points.keys()) / point_num_threshold)*100
      print(f'running iteration {interation}: {percent:.3}‰')
      line_model, inliers = ransac(remaining_edges, LineModelND, min_samples=3, residual_threshold=line_threshold, max_trials=1000)

      # Store found edges
      inlier_points = remaining_edges[inliers]
      for point in inlier_points:
        key = str(point[0]) + "," + str(point[1]) + "," + str(point[2])
        if key in found_edge_points.keys():
          found_edge_points[key] += 1
        else:
          found_edge_points[key] = 1
      
      # Remove inliers from remaining_edges
      remaining_edges = remaining_edges[inliers == False]

      # Update iteration variable
      interation += 1
    
    print(len(found_edge_points.keys()), "/", edges.shape[0])
    # Go through each line model
      # count each of the points close to the line up one
    
    # Give colors to the points that have more than one line on them

    # Visualize





if __name__ == "__main__":
  main()