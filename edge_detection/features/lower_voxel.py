import numpy as np
import open3d as o3d
from feature import VoxelFeature


# OBS: voxel.grid_index !== voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

# TODO: Must segment some roofs into different types of edges and corners.
#       Then I need to calculate many different neighborhood matrises and average over the segmented values

# Edge if:
# * No voxels below in neighborhood 
# * Max 3 voxels in same z neighborhood
class LowerVoxel(VoxelFeature):
  def run_at_scale(self, scale=float, visualize=True):
    all_voxels = self.voxel_grid.get_voxels()
    voxels = np.asarray(all_voxels)
    labels = np.ones(self.points.shape[0])*-1 # Default to not a upper_voxel

    for voxel_i in range(voxels.shape[0]):
      # Check if there are neighboring voxels straight or diagonally below.
      grid_index = voxels[voxel_i].grid_index
      center = self.voxel_grid.get_voxel_center_coordinate(grid_index)

      # All neighbors below current voxel
      below_neighbors = np.zeros((3*3*2*2, 3))
      n = 0
      for x in [-1,0,1]:
        for y in [-1,0,1]:
          for h in range(1, 3):
            for s in range(1, 3):
              point = np.array([center[0]+scale*x*s, center[1]+scale*y*s, center[2]-scale*h])
              below_neighbors[n] = point
              n += 1

      below_query = o3d.utility.Vector3dVector(below_neighbors)

      # Visualize below_query
      # pcd = o3d.geometry.PointCloud()
      # pcd.points = o3d.utility.Vector3dVector(below_query)
      # pcd.paint_uniform_color([1,0,0])
      # o3d.visualization.draw([self.voxel_grid, pcd])

      below_included = self.voxel_grid.check_if_included(below_query)

      # If they are a upper_voxel, store it in the labels
      if np.sum(below_included) == 0:
        point_indices = self.grid_index_to_point_indices[tuple(grid_index)]
        labels[point_indices] = 1

    # Visualize colored voxels
    if visualize:
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

  # print("Processing", file_name)

  cloud = read_roof_cloud(file_name)

  # o3d.visualization.draw_geometries([cloud])

  print(cloud)

  f = LowerVoxel(cloud)

  project_folder = get_project_folder()
  image_folder = os.path.join(project_folder, 'edge_detection/results/feature/lower_voxel/images/' + file_name_base + '/')
  
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