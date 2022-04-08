import numpy as np
import open3d as o3d
from features.feature import VoxelFeature, ScalableFeatureState


# OBS: voxel.grid_index !== voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

# TODO: Must segment some roofs into different types of edges and corners.
#       Then I need to calculate many different neighborhood matrises and average over the segmented values

# Edge if no voxels above in neighborhood 
H = 3
K = 2*H + 1
class UpperVoxels(VoxelFeature):
  def run_at_scale(self, scale=float, visualize=True):
    all_voxels = self.voxel_grid.get_voxels()
    voxels = np.asarray(all_voxels)
    labels = np.ones(self.state.points.shape[0])*-1 # Default to not a upper_voxel

    for voxel_i in range(voxels.shape[0]):
      # Check if there are neighboring voxels straight or diagonally above.
      grid_index = voxels[voxel_i].grid_index
      voxel_center = self.voxel_grid.get_voxel_center_coordinate(grid_index)

      # All neighbors above current voxel
      neighbor_point = lambda x, y: [voxel_center[0]+scale*x, voxel_center[1]+scale*y, voxel_center[2]+scale]
      above_neighbors = np.array([neighbor_point(x, y) for x in range(-H, H+1) for y in range(-H, H+1)])

      above_query = o3d.utility.Vector3dVector(above_neighbors)

      # Visualize above_query
      # pcd = o3d.geometry.PointCloud()
      # pcd.points = o3d.utility.Vector3dVector(above_query)
      # pcd.paint_uniform_color([1,0,0])
      # o3d.visualization.draw([self.voxel_grid, pcd])

      above_included = self.voxel_grid.check_if_included(above_query)

      # If they are a upper_voxel, store it in the labels
      if np.sum(above_included) == 0:
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
  from helpers import read_roof_cloud, normalize_cloud
  file_name = "32-1-510-215-53-test-1.ply"
  cloud = read_roof_cloud(file_name)
  cloud = normalize_cloud(cloud)
  print(cloud)

  state = ScalableFeatureState(cloud)
  f = UpperVoxels(state)
  f.run_test('upper_voxel', file_name)
