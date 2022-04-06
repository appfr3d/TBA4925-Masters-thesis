import numpy as np
import open3d as o3d
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from feature import SmallVoxelFeature

H = 2
K = 2*H + 1

class LowerVoxel(SmallVoxelFeature):
  def run_at_scale(self, scale=float, visualize=True):
    all_voxels = self.voxel_grid.get_voxels()
    voxels = np.asarray(all_voxels)
    labels = np.ones(self.points.shape[0])*-1 # Default to not a upper_voxel

    # Smallest, middle and largest lists
    smallest = np.zeros(self.points.shape[0])
    middle = np.zeros(self.points.shape[0])
    largest = np.zeros(self.points.shape[0])

    # Make a new voxel list with all the occupied voxels
    # Make a new empty dict called full
    # For each occupied voxel:
    #   if not in dictadd to dict
    #   for each neighbor of the voxel
    #     if not in dict add to dict
    voxels_with_neighbors = {}
    for voxel_i in range(voxels.shape[0]):
      grid_index = tuple(voxels[voxel_i].grid_index)
      
      for x in range(-H, H+1):
        for y in range(-H, H+1):
          for z in range(-H, H+1):
            neighbor_grid_index = (grid_index[0] + x, grid_index[1] + y, grid_index[2] + z)
            if not neighbor_grid_index in voxels_with_neighbors.keys():
              voxels_with_neighbors[neighbor_grid_index] = np.zeros(6)
              # Maybe move logic in here?? then we can use x, y, z....

    # For each voxel in the full-dict
    #   Calculate Ix, Iy, Iz
    #   Calculate Ixx, Ixy, etc
    #   Store as [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
    for grid_index in voxels_with_neighbors.keys():
      x_part = np.power(grid_index[0], 2) / np.power(H/2, 2)
      y_part = np.power(grid_index[1], 2) / np.power(H/2, 2)
      z_part = np.power(grid_index[2], 2) / np.power(H/2, 2)

      # print('x_part, y_part, z_part', x_part, y_part, z_part)
      # OBS: bug with exponent. Values become 0 or inf
      exponent = np.exp(-(x_part + y_part + z_part)) # 1 # x_part + y_part + z_part # 

      # OBS: Not sure to multiply with grid_index or not here...
      Ix = - grid_index[0] * exponent
      Iy = - grid_index[1] * exponent
      Iz = - grid_index[2] * exponent

      # print('Ix, Iy, Iz', Ix, Iy, Iz)

      # Stored as [Ixx, Iyy, Izz, Ixy, Ixz, Iyz] 
      voxels_with_neighbors[grid_index] = np.array([Ix*Ix, Iy*Iy, Iz*Iz, Ix*Iy, Ix*Iz, Iy*Iz]) 

    # For all occupied voxels v
    #   Calculate Axx = G*Ix^2, Axy=G*IxIy, etc
    #   Calculate A(v) matrix
    #   Calculate eigen values for A(v) matrix, and sort
    #   Classify voxel v based on eigen values
    for voxel_i in range(voxels.shape[0]):
      grid_index = tuple(voxels[voxel_i].grid_index)
      
      neighborhood_I = np.zeros((K, K, K, 6))
      for x in range(-H, H+1):
        for y in range(-H, H+1):
          for z in range(-H, H+1):
            neighbor_grid_index = (grid_index[0] + x, grid_index[1] + y, grid_index[2] + z)
            neighborhood_I[x, y, z] = voxels_with_neighbors[neighbor_grid_index]

      # Stored as [Axx, Ayy, Azz, Axy, Axz, Ayz]
      structure_tensor_values = np.zeros(6)
      for i in range(6):
        g = gaussian_filter(neighborhood_I[:,:,:,i], sigma=1)
        # print(g)
        # print(type(g))
        # print(g.shape)
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

      # TODO: refine this condition
      # condition = smallest[voxel_i] < 0.1 and middle[voxel_i] > 0.1
      condition = smallest[voxel_i] * 10 < middle[voxel_i]
      # print('condition:', condition)
      if condition:
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

    fig, axs = plt.subplots(1, 5)
    fig.suptitle('Histogram of smallest, middle and largest eigen values')

    axs[0].hist(smallest)
    axs[0].set_title('Smallest')

    axs[1].hist(middle)
    axs[1].set_title('Middle')

    axs[2].hist(largest)
    axs[2].set_title('Largest')

    ratio = middle/largest
    axs[3].hist(ratio)
    axs[3].set_title('All ratio')

    chosen_ratio = ratio[np.bitwise_and(smallest < 0.1, middle > 0.1)]
    axs[4].hist(chosen_ratio)
    axs[4].set_title('Chosen ratio')
    
    plt.show()


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
  image_folder = os.path.join(project_folder, 'edge_detection/results/feature/edge_voxel/images/' + file_name_base + '/')
  
  # Create folder if not exists
  if not os.path.exists(image_folder):
    os.makedirs(image_folder)

  print('Start running...')

  all_labels = f.run()

  print('Done running! Start saving...', end='')

  # Create window
  vis = o3d.visualization.Visualizer()
  vis.create_window(width=1000, height=1000)
  vis.add_geometry(cloud)

  for label_i in range(all_labels.shape[0]):
    labels = all_labels[label_i]
    save_scaled_feature_image(vis, cloud, labels, image_folder, str(label_i))


  labels = np.sum(all_labels, axis=0)

  save_scaled_feature_image(vis, cloud, labels, image_folder, "Combined")

  print('done!')

  vis.destroy_window()