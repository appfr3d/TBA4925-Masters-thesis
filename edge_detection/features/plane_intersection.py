import open3d as o3d
import copy
from features.feature import Feature
from features.helpers import plane_intersect, point_to_line_distance

class PlaneIntersection(Feature):
  def run(self):
    planes = []
    rest = copy.deepcopy(self.cloud)

    # TODO: calculate best distance threshold here...
    threshold = 0.03

    while len(rest.points) > 100:
      # Calculate best fitted plane using RANSAC
      plane_model, inliers = rest.segment_plane(distance_threshold=threshold, ransac_n=3, num_iterations=100)
      # print('plane_model', plane_model)

      # Save best fitted plane
      planes.append(plane_model)

      # Remove points in plane from rest
      rest = rest.select_by_index(inliers, invert=True)

    plane_intersection_lines = []
    for p_i in range(len(planes)):
      for p_j in range(p_i+1, len(planes)):
        # calculate intersection line
        line = plane_intersect(planes[p_i], planes[p_j])
        plane_intersection_lines.append(line)

    plane_intersection_point_indices = {} # point_i: number of close lines 
    i = 1
    for line in plane_intersection_lines:
      # print('plane number:', i)
      i += 1
      for point_i, point in enumerate(self.cloud.points):
        # calculate distance from point to intersection line
        # TODO: revwrite to "cloud_to_line_distance" for better performance if needed
        dist = point_to_line_distance(line[0], line[1], point)
        if dist < threshold:
          if point_i in plane_intersection_point_indices:
            plane_intersection_point_indices[point_i] += 1
          else:
            plane_intersection_point_indices[point_i] = 1

    # Return a dict of indices of point that are close to the intersection of planes fitted in the point cloud, 
    # and the number of intersections the points are close to:
    # point_i: number of close lines 
    return plane_intersection_point_indices

if __name__ == "__main__":
  from helpers import read_roof_cloud
  file_name = "32-1-510-215-53-roof-2-shift.ply"
  cloud = read_roof_cloud(file_name)

  f = PlaneIntersection(cloud)

  intersecting_points = f.run()
  intersecting_points_indices = intersecting_points.keys()

  edge_indices = []
  corner_indices = []
  for point_i in intersecting_points_indices:
    if intersecting_points[point_i] > 1:
      corner_indices.append(point_i)
    else:
      edge_indices.append(point_i)

  edge_points = cloud.select_by_index(edge_indices)
  corner_points = cloud.select_by_index(corner_indices)
  other = cloud.select_by_index(list(intersecting_points_indices), invert=True)

  edge_points.paint_uniform_color([0, 1, 0])    # green
  corner_points.paint_uniform_color([1, 0, 0])  # red
  other.paint_uniform_color([0.6, 0.6, 0.6])    # gray

  o3d.visualization.draw_geometries([edge_points, corner_points, other])
