import numpy as np
import open3d as o3d
from feature import ScalableFeature
from matplotlib import pyplot as plt


#####
#       OBS: must normalize point clouds before saving them to use this feature!
#             or else we will loose a lot of precision...
#####
class CovarianceEigenvalue(ScalableFeature):
  def preprocess_whole_cloud(self):
    # Create KDTree
    self.kd_tree = o3d.geometry.KDTreeFlann(self.cloud)

  def run_at_scale(self, scale: float):
    points = np.asarray(self.cloud.points)
    labels = np.zeros(points.shape[0])
    covariances = np.asarray(o3d.geometry.PointCloud.estimate_point_covariances(self.cloud, o3d.geometry.KDTreeSearchParamRadius(scale)))
    
    # Smallest, middle and largest lists
    smallest = np.zeros(points.shape[0])
    middle = np.zeros(points.shape[0])
    largest = np.zeros(points.shape[0])

    # Run through every point
    for point_i, point in enumerate(points):
      # Find eigenvalues of the points covariance matrix
      [eigen_values, eigen_vectors] = np.linalg.eig(covariances[point_i])


      # We expect edges to have two large and one small eigen value
      sigma = lambda eigen: eigen[0]/np.sum(eigen)

      labels[point_i] = sigma(eigen_values) # eigen_values[0]


      # Store smallest, middle and largest eigen values
      eigen_values_sorted = np.sort(eigen_values)


      smallest[point_i] = eigen_values_sorted[0]
      middle[point_i] = eigen_values_sorted[1]
      largest[point_i] = eigen_values_sorted[2]

      # labels[1, point_i] = eigen_values[1]
      # labels[2, point_i] = eigen_values[2]

      '''
      y=Math.atan(k*x)/(Math.PI/2);
      y=x/(k+x);
      y=1-Math.exp(-k*x);
      
      k = 1
      scale = lambda eigen: np.arctan(k*eigen)/np.pi*0.5

      lambda_0_mapped = scale(eigen_values[0])
      lambda_1_mapped = scale(eigen_values[1])
      lambda_2_mapped = scale(eigen_values[2])

      labels[0, point_i] = lambda_0_mapped
      labels[1, point_i] = lambda_1_mapped
      labels[2, point_i] = lambda_2_mapped
      '''
    
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

    chosen_ratio = ratio[np.bitwise_and(smallest < 0.1, largest > 0.1)]
    axs[4].hist(chosen_ratio)
    axs[4].set_title('Chosen ratio')
    
    plt.show()


    return labels

if __name__ == "__main__":
  import os
  from helpers import read_roof_cloud, get_project_folder, save_scaled_feature_image
  file_name_base = "32-1-510-215-53-test-2"
  file_name = file_name_base + ".ply"
  cloud = read_roof_cloud(file_name)

  f = CovarianceEigenvalue(cloud)
  all_labels = f.run()
  print('all_labels.shape', all_labels.shape)

  project_folder = get_project_folder()
  image_folder = os.path.join(project_folder, 'edge_detection/results/feature/covariance_eigenvalue/images/' + file_name_base + '/')
  
  # Create folder if not exists
  if not os.path.exists(image_folder):
    os.makedirs(image_folder)

  # Threshold
  threshold = [0.03, 0.02, 0.02]

  # Create window
  vis = o3d.visualization.Visualizer()
  vis.create_window(width=1000, height=1000)
  vis.add_geometry(cloud)

  labels_corrected = np.zeros(all_labels.shape)

  for label_i in range(all_labels.shape[0]):
    labels = all_labels[label_i] # - threshold[label_i]
    max_l = np.max(labels)
    min_l = np.min(labels)

    # k = 1
    # (1/max_l)*eigen
    scale = lambda eigen: (1/max_l)*eigen
    # np.arctan(k*eigen)/np.pi*0.5
    labels_scaled = scale(labels)
    # plt.scatter(list(range(labels.shape[0])), labels, c='r')
    # plt.scatter(list(range(labels.shape[0])), labels_scaled, c='b')
    # plt.xlabel('index')
    # plt.ylabel('λ0')
    # plt.title('Scale' + str(label_i))
    # plt.show()
    # Visualize
    # colors = np.zeros((labels.shape[0], 3))
    # colors += [0.6, 0.6, 0.6]
    # colors[labels >= 0] = [0, 1, 0] # Color positive values as green
    # colors[:, 0] = labels_scaled # Color based on eigen value
    # colors[:, 1] = labels_scaled
    # colors[:, 2] = labels_scaled
    # cloud.colors = o3d.utility.Vector3dVector(colors[:, :3])
    # o3d.visualization.draw_geometries([cloud])


    # Calculate threshold
    labels_sorted = np.sort(labels_scaled)
    largest_gap = [0, 0] # value, index
    for i in range(labels_sorted.shape[0] - 1):
      gap = labels_sorted[i+1] - labels_sorted[i]

      # There can be noise in the dataset, so only look at the 95% first
      if gap > largest_gap[0] and i < labels_sorted.shape[0]*0.95:
        largest_gap = [gap, i]

    # Threshold in the middle of the largest gap
    threshold = labels_sorted[largest_gap[1]] + largest_gap[0]*0.5
    print('treshold', threshold)

    # Visualize
    # plt.scatter(list(range(labels.shape[0])), labels_sorted)
    # plt.xlabel('index')
    # plt.ylabel('σ')
    # plt.title('Scale' + str(label_i))
    # plt.show()

    labels_real = labels_scaled - threshold
    labels_corrected[label_i] = labels_real 
    save_scaled_feature_image(vis, cloud, labels_real, image_folder, str(label_i))

  labels = np.sum(labels_corrected, axis=0)
  save_scaled_feature_image(vis, cloud, labels, image_folder, "Combined")

  vis.destroy_window()
  '''
  file_out_name = "32-1-510-215-53-normal_cluster-shift.ply"
  write_roof_cloud_result(file_name, cloud)
  '''


