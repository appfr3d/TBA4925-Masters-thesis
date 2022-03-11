import numpy as np
import open3d as o3d
from feature import VoxelFeature


# OBS: voxel.grid_index !== voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

# TODO: Must segment some roofs into different types of edges and corners.
#       Then I need to calculate many different neighborhood matrises and average over the segmented values

# Edge if:
# * No voxels above in neighborhood 
# * Max 3 voxels in same z neighborhood
class UpperVoxel(VoxelFeature):
  def run_at_sacale(self, scale=float):
    all_voxels = self.voxel_grid.get_voxels()
    voxels = np.asarray(all_voxels)
    labels = np.ones(self.points.shape[0])*-1 # Default to not a upper_voxel
    # labels = np.zeros(voxels.shape[0])

    # max_z = np.max([v.grid_index[2] for v in all_voxels])

    # Visualize voxels at different scales
    # o3d.visualization.draw([self.voxel_grid])

    for voxel_i in range(voxels.shape[0]):
      # if voxels[voxel_i].grid_index[2] == max_z:
      # Check if there are neighboring voxels straight or diagonally above.
      grid_index = voxels[voxel_i].grid_index
      center = self.voxel_grid.get_voxel_center_coordinate(grid_index)

      # All neighbors above current voxel
      above_neighbors = np.zeros((9, 3))
      n = 0
      for x in [-1,0,1]:
        for y in [-1,0,1]:
            point = np.array([center[0]+scale*x, center[1]+scale*y, center[2]+scale])
            above_neighbors[n] = point
            n += 1

      around_neighbors = np.zeros((8, 3))
      n = 0
      for x in [-1,0,1]:
        for y in [-1,0,1]:
          if not (x == 0 and y == 0):
            point = np.array([center[0]+scale*x, center[1]+scale*y, center[2]])
            around_neighbors[n] = point
            n += 1
      


      above_query = o3d.utility.Vector3dVector(above_neighbors)
      around_query = o3d.utility.Vector3dVector(around_neighbors)

      # Visualize above_query
      # pcd = o3d.geometry.PointCloud()
      # pcd.points = o3d.utility.Vector3dVector(above_query)
      # pcd.paint_uniform_color([1,0,0])
      # o3d.visualization.draw([self.voxel_grid, pcd])

      above_included = self.voxel_grid.check_if_included(above_query)
      around_included = self.voxel_grid.check_if_included(around_query)

      # If they are a upper_voxel, store it in the labels
      if np.sum(above_included) == 0 and np.sum(around_included) <= 3:
        # Find points in circle around the voxel
        radius = scale * np.sqrt(2) * 0.5
        [_, idx, _] = self.kd_tree.search_radius_vector_3d(center, radius)

        # Exclude the points not in the square ([]) <-
        bound = np.asarray(self.voxel_grid.get_voxel_bounding_points(grid_index))
        bound_x = [np.max(bound[:, 0]), np.min(bound[:, 0])]
        bound_y = [np.max(bound[:, 1]), np.min(bound[:, 1])]
        bound_z = [np.max(bound[:, 2]), np.min(bound[:, 2])]
        true_idx = []
        for point_i in idx:
          if  bound_x[0] >= self.points[point_i][0] >= bound_x[1] and \
              bound_y[0] >= self.points[point_i][1] >= bound_y[1] and \
              bound_z[0] >= self.points[point_i][2] >= bound_z[1]:
            true_idx.append(point_i)

        if len(true_idx) == 0:
          print('true_idx len == 0... wtf, idx len =', len(idx))   
        labels[true_idx] = 1

    # Visualize colored voxels
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(self.points)
    colors = np.asarray(self.cloud.colors)
    colors[labels >= 0] = [0, 1, 0] # Color positive values as green
    pcd.colors = o3d.utility.Vector3dVector(colors)

    colored_voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=scale)
    o3d.visualization.draw([colored_voxels])

    return labels 


if __name__ == "__main__":
  import os
  from helpers import read_roof_cloud, get_project_folder, save_scaled_feature_image
  file_name_base = "32-1-510-215-53-test-1"
  file_name = file_name_base + ".ply"
  cloud = read_roof_cloud(file_name)

  # o3d.visualization.draw_geometries([cloud])

  f = UpperVoxel(cloud)

  project_folder = get_project_folder()
  image_folder = os.path.join(project_folder, 'edge_detection/results/feature/upper_voxel/images/' + file_name_base + '/')
  
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