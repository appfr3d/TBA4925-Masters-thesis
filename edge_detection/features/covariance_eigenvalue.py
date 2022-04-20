import numpy as np
import open3d as o3d
from features.feature import ScalableFeature, ScalableFeatureState
from matplotlib import pyplot as plt


#####
#       OBS: must normalize point clouds before saving them to use this feature!
#             or else we will loose a lot of precision...
#####
class CovarianceEigenvalue(ScalableFeature):
  def run_at_scale(self, scale: float):
    labels = np.zeros(self.state.points.shape[0])
    covariances = np.asarray(o3d.geometry.PointCloud.estimate_point_covariances(self.state.cloud, o3d.geometry.KDTreeSearchParamRadius(scale)))
    
    # Smallest, middle and largest lists
    # smallest = np.zeros(self.state.points.shape[0])
    # middle = np.zeros(self.state.points.shape[0])
    # largest = np.zeros(self.state.points.shape[0])

    # Run through every point
    for point_i in range(self.state.points.shape[0]):
      # Find eigenvalues of the points covariance matrix
      [eigen_values, eigen_vectors] = np.linalg.eig(covariances[point_i])

      # Store smallest, middle and largest eigen values
      eigen_values_sorted = np.sort(eigen_values)

      # We expect edges to have two large and one small eigen value
      sigma = lambda eigen: eigen[0]/np.sum(eigen)
      labels[point_i] = sigma(eigen_values_sorted)

      # smallest[point_i] = eigen_values_sorted[0]
      # middle[point_i] = eigen_values_sorted[1]
      # largest[point_i] = eigen_values_sorted[2]

    # Post process to correct labales
    max_l = np.max(labels)

    # Other scaling options
    # (1/max_l)*eigen
    # k = 1
    # np.arctan(k*eigen)/np.pi*0.5
    scale_fn = lambda eigen: (1/max_l)*eigen
    labels_scaled = scale_fn(labels)

    # Calculate threshold
    # labels_sorted = np.sort(labels_scaled)
    # largest_gap = [0, 0] # value, index
    # for i in range(labels_sorted.shape[0] - 1):
    #   gap = labels_sorted[i+1] - labels_sorted[i]

    #   # There can be noise in the dataset, so only look at the 95% first
    #   if gap > largest_gap[0] and i < labels_sorted.shape[0]*0.95:
    #     largest_gap = [gap, i]

    # # Threshold in the middle of the largest gap
    # self.thresholds[self.state.scales.index(scale)] = labels_sorted[largest_gap[1]] + largest_gap[0]*0.5
    # threshold = labels_sorted[largest_gap[1]] + largest_gap[0]*0.5
    # print('treshold', threshold)

    # labels_real = labels_scaled - threshold

    return labels_scaled

if __name__ == "__main__":
  from helpers import read_roof_cloud, normalize_cloud
  file_name = "32-1-510-215-53-test-1.ply"
  cloud = read_roof_cloud(file_name)
  cloud = normalize_cloud(cloud)
  print(cloud)

  state = ScalableFeatureState(cloud)
  f = CovarianceEigenvalue(state)
  f.run_test('covariance_eigenvalue', file_name)



