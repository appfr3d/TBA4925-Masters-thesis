import numpy as np
import open3d as o3d
from features.feature import VoxelFeature, ScalableFeatureState


# OBS: voxel.grid_index !== voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

# TODO: Must segment some roofs into different types of edges and corners.
#       Then I need to calculate many different neighborhood matrises and average over the segmented values

# Edge if no voxels below in neighborhood 
H = 3
K = 2*H + 1

class LowerVoxels(VoxelFeature):
  def run_at_scale(self, scale=float, knn_scale=int, visualize=True):
    all_voxels = self.voxel_grid.get_voxels()
    voxels = np.asarray(all_voxels)
    labels = np.zeros(self.state.points.shape[0]) # Default to not a upper_voxel

    for voxel_i in range(voxels.shape[0]):
      # Check if there are neighboring voxels straight or diagonally below.
      grid_index = voxels[voxel_i].grid_index
      voxel_center = self.voxel_grid.get_voxel_center_coordinate(grid_index)

      # All neighbors below current voxel
      neighbor_point = lambda x, y: [voxel_center[0]+scale*x, voxel_center[1]+scale*y, voxel_center[2]-scale]
      below_neighbors = np.array([neighbor_point(x, y) for x in range(-H, H+1) for y in range(-H, H+1)])

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
      pcd.points = o3d.utility.Vector3dVector(self.state.points)
      colors = np.asarray(self.state.cloud.colors)
      colors[labels > 0] = [0, 1, 0] # Color positive values as green
      pcd.colors = o3d.utility.Vector3dVector(colors)

      colored_voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=scale)
      o3d.visualization.draw([colored_voxels])

    return labels 


if __name__ == "__main__":
  from helpers import read_roof_cloud, normalize_cloud
  file_name = "32-1-510-215-53-test-1.ply"
  cloud = read_roof_cloud(file_name)
  cloud = normalize_cloud(cloud)
  print(cloud)

  state = ScalableFeatureState(cloud)
  f = LowerVoxels(state)
  f.run_test('lower_voxel', file_name)