import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from features.feature import SmallVoxelFeature, SmallScalableFeatureState

H = 2
K = 2*H + 1

class EdgeVoxels(SmallVoxelFeature):
  def run_at_scale(self, scale:float, knn_scale:int, visualize=True):
    all_voxels = self.voxel_grid.get_voxels()
    voxels = np.asarray(all_voxels)
    labels = np.ones(self.state.points.shape[0])*-1 # Default to not a upper_voxel

    # Smallest, middle and largest eigen value lists
    smallest = np.zeros(self.state.points.shape[0])
    middle = np.zeros(self.state.points.shape[0])
    largest = np.zeros(self.state.points.shape[0])

    # Make a new voxel list with all the occupied voxels
    # Make a new empty dict called full
    # For each occupied voxel:
    #   if not in dictadd to dict
    #   for each neighbor of the voxel
    #     if not in dict add to dict

    max_idx = np.array([-100, -100, -100])
    for voxel_i in range(voxels.shape[0]):
      grid_index = tuple(voxels[voxel_i].grid_index)
      for i in range(3):
        if max_idx[i] < grid_index[i]:
          max_idx[i] = grid_index[i]

    max_idx = max_idx + K # Padding

    occupancy_grid = np.zeros(max_idx, dtype=np.float32)

    for voxel_i in range(voxels.shape[0]):
      grid_index = tuple(voxels[voxel_i].grid_index)
      # Add occupied voxels
      occupancy_grid[grid_index] = 1.0

    # Presmooth occupancy_grid
    occupancy_grid = gaussian_filter(occupancy_grid, sigma=1)

    # Store exponent * presmoothed occupancy_grid in voxels_with_neighbors
    voxels_with_neighbors = {}
    for voxel_i in range(voxels.shape[0]):
      grid_index = tuple(voxels[voxel_i].grid_index)

      voxel_center = self.voxel_grid.get_voxel_center_coordinate(grid_index)
      
      for x in range(-H, H+1):
        for y in range(-H, H+1):
          for z in range(-H, H+1):
            neighbor_grid_index = (grid_index[0] + x, grid_index[1] + y, grid_index[2] + z)
            if not neighbor_grid_index in voxels_with_neighbors.keys():
              # voxels_with_neighbors[neighbor_grid_index] = np.zeros(6)

              # Normalized coordinates
              [Xp, Yp, Zp] = voxel_center + [x*scale, y*scale, z*scale]
              
              x_part = np.power(Xp, 2) / np.power(H/2, 2)
              y_part = np.power(Yp, 2) / np.power(H/2, 2)
              z_part = np.power(Zp, 2) / np.power(H/2, 2)

              exponent = np.exp(-(x_part + y_part + z_part))
              voxels_with_neighbors[neighbor_grid_index] = exponent * occupancy_grid[neighbor_grid_index]

    for voxel_i in range(voxels.shape[0]):
      grid_index = tuple(voxels[voxel_i].grid_index)

      I_neighborhood = np.zeros((K, K, K, 6))
      for x in range(-H, H+1):
        for y in range(-H, H+1):
          for z in range(-H, H+1):
            neighbor_grid_index = (grid_index[0] + x, grid_index[1] + y, grid_index[2] + z)

            # Shift
            [Xs, Ys, Zs] = [x*scale, y*scale, z*scale]

            exponent = voxels_with_neighbors[neighbor_grid_index]

            Ix = - Xs * exponent
            Iy = - Ys * exponent
            Iz = - Zs * exponent

            I_neighborhood[x, y, z] = np.array([Ix*Ix, Iy*Iy, Iz*Iz, Ix*Iy, Ix*Iz, Iy*Iz]) 

      # Stored as [Axx, Ayy, Azz, Axy, Axz, Ayz]
      structure_tensor_values = np.zeros(6)
      for i in range(6):
        g = gaussian_filter(I_neighborhood[:,:,:,i], sigma=1)
        structure_tensor_values[i] = g[H, H, H]

      structure_tensor = np.array([
        [structure_tensor_values[0], structure_tensor_values[3], structure_tensor_values[4]], 
        [structure_tensor_values[3], structure_tensor_values[1], structure_tensor_values[5]], 
        [structure_tensor_values[4], structure_tensor_values[5], structure_tensor_values[2]]])

      [eigen_values, eigen_vectors] = np.linalg.eig(structure_tensor)

      # Store smallest, middle and largest eigen values
      eigen_values_sorted = np.sort(eigen_values)

      # print('eigen_values_sorted', eigen_values_sorted)

      smallest[voxel_i] = eigen_values_sorted[0]
      middle[voxel_i] = eigen_values_sorted[1]
      largest[voxel_i] = eigen_values_sorted[2]

      point_indices = self.grid_index_to_point_indices[tuple(grid_index)]
      # print("condition value:", smallest[voxel_i] / middle[voxel_i])
      labels[point_indices] = smallest[voxel_i] / middle[voxel_i]

    # Post process to correct labales
    max_l = np.max(labels)
    scale_fn = lambda val: (1/max_l)*val
    labels_scaled = scale_fn(labels)

    # Visualize colored voxels
    if visualize:
      pcd = o3d.geometry.PointCloud()
      pcd.points = o3d.utility.Vector3dVector(self.state.points)
      colors = np.asarray(self.state.cloud.colors)
      colors[labels >= 0] = [0, 1, 0] # Color positive values as green
      pcd.colors = o3d.utility.Vector3dVector(colors)

      colored_voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=scale)
      o3d.visualization.draw([colored_voxels])

      fig, axs = plt.subplots(1, 4)
      fig.suptitle('Histogram of smallest, middle and largest eigen values')

      axs[0].hist(smallest)
      axs[0].set_title('Smallest')

      axs[1].hist(middle)
      axs[1].set_title('Middle')

      axs[2].hist(largest)
      axs[2].set_title('Largest')

      ratio = smallest/middle
      axs[3].hist(ratio)
      axs[3].set_title('All ratio')
      
      plt.show()

    return labels_scaled 


if __name__ == "__main__":
  from helpers import read_roof_cloud, normalize_cloud
  file_name = "32-1-510-215-53-test-3.ply"
  cloud = read_roof_cloud(file_name)
  cloud = normalize_cloud(cloud)
  print(cloud)

  state = SmallScalableFeatureState(cloud)
  f = EdgeVoxels(state)
  f.run_test('edge_voxel', file_name)