import copy
import numpy as np
import open3d as o3d
from features.feature import VoxelFeature, ScalableFeatureState

# OBS: voxel.grid_index !== voxel_grid.get_voxel_center_coordinate(voxel.grid_index)

# TODO: Must segment some roofs into different types of edges and corners.
#       Then I need to calculate many different neighborhood matrises and average over the segmented values

# Edge if there are voxels on one site but not the other side
H = 2 # number of x,y neighbors on each side
K = 2*H + 1
L = 4 # length to go up and down in z neighbors
Z = 2*L + 1

outer_sides = [[0,1,2,3,4], [4,9,13,18,23], [19,20,21,22,23], [0,5,10,14,29]]
inner_sides = [[6,7,8], [8,12,17], [15,16,17], [6,11,15]]
class AroundVoxels(VoxelFeature):
  def run_at_scale(self, scale:float, knn_scale:int, visualize=True):
    all_voxels = self.voxel_grid.get_voxels()
    voxels = np.asarray(all_voxels)
    labels = np.zeros(self.state.points.shape[0]) # Default to not a around_voxel

    # Evaluate each voxel
    for voxel_i in range(voxels.shape[0]):
      grid_index = voxels[voxel_i].grid_index
      voxel_center = self.voxel_grid.get_voxel_center_coordinate(grid_index)

      # All voxels surrounding current voxel
      around_neighbors = np.zeros((K, K, Z, 3))
      for x in range(-H, H+1):
        for y in range(-H, H+1):
          # We don not to consider straight above or beneeth
          if not (x == 0 and y == 0):
            for z in range(-L, L+1):
              around_neighbors[x, y, z] = np.array([voxel_center[0]+scale*x, voxel_center[1]+scale*y, voxel_center[2]+z*scale])

      reshaped_around = around_neighbors.reshape((K*K*Z, 3))

      around_query = o3d.utility.Vector3dVector(reshaped_around)

      # Visualize above_query
      # pcd = o3d.geometry.PointCloud()
      # pcd.points = around_query
      # pcd.paint_uniform_color([1,0,0])
      # o3d.visualization.draw([self.voxel_grid, pcd])

      around_included = np.array(self.voxel_grid.check_if_included(around_query)).reshape((K, K, Z))

      directions = np.full((4, H), False)
      for s in range(1, H+1):
        # or cus we want any true value in a column to count
        N = np.any(around_included[0, -s])
        S = np.any(around_included[0, s])
        W = np.any(around_included[-s, 0])
        E = np.any(around_included[s, 0])

        NE = np.any(around_included[s, -s])
        SE = np.any(around_included[s, s])

        NW = np.any(around_included[-s, -s])
        SW = np.any(around_included[-s, s])

        # 0:vertical, 1:horizontal, 2:up-right, 3:down-right
        # xor cus we want one side to be included, and another to not be
        directions[0, s-1] = np.bitwise_xor(N, S)
        directions[1, s-1] = np.bitwise_xor(W, E)
        directions[2, s-1] = np.bitwise_xor(NE, SW)
        directions[3, s-1] = np.bitwise_xor(SE, NW)

      # Merge directions along H-scale
      directions = np.all(directions, axis=1)

      # If they are a upper_voxel, store it in the labels
      num_direction = np.sum(directions)
      if num_direction >= 1:
        point_indices = self.grid_index_to_point_indices[tuple(grid_index)]
        labels[point_indices] = num_direction / directions.shape[0]

    # Visualize colored voxels
    if visualize:
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(self.state.points)
      colors = np.asarray(copy.deepcopy(self.state.cloud.colors))
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
  f = AroundVoxels(state)
  f.run_test('around_voxel', file_name)