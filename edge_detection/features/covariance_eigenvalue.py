import numpy as np
import math
import open3d as o3d
from features.feature import ScalableMultiFeature, ScalableFeatureState
from matplotlib import pyplot as plt


#####
#       OBS: must normalize point clouds before saving them to use this feature!
#             or else we will loose a lot of precision...
#####
class CovarianceEigenvalue(ScalableMultiFeature):
  def run_at_scale(self, scale: float, knn_scale: int):
    labels = np.zeros((8, self.state.points.shape[0]))
    covariances = np.asarray(o3d.geometry.PointCloud.estimate_point_covariances(self.state.cloud, o3d.geometry.KDTreeSearchParamRadius(scale)))
    
    # Implementing functions for the 8 features proposed by Weinmann et. al.
    eigen_sum = lambda eigen: np.sum(eigen)
    omnivariance = lambda eigen: np.cbrt(np.prod(eigen))
    eigenentropy = lambda eigen: -np.sum(eigen*np.log2(np.abs(eigen)))
    anisotropy = lambda eigen: (eigen[2] - eigen[0]) / eigen[2]
    planarity = lambda eigen: (eigen[1] - eigen[0]) / eigen[2]
    linearity = lambda eigen: (eigen[2] - eigen[1]) / eigen[2]
    surface_variation = lambda eigen: eigen[0]/np.sum(eigen)
    sphericity = lambda eigen: eigen[0]/eigen[2]

    # Run through every point
    for point_i in range(self.state.points.shape[0]):
      # Find eigenvalues of the points covariance matrix
      [eigen_values, eigen_vectors] = np.linalg.eig(covariances[point_i])

      # Store smallest, middle and largest eigen values
      eigen_values_sorted = np.sort(eigen_values)
      if eigen_values_sorted[0] == 0:
        eigen_values_sorted[0] += 1.0e-20

      # Run and store all functions
      # [scale, feature, point_index]
      labels[0, point_i] = eigen_values_sorted[0]
      labels[1, point_i] = eigen_values_sorted[1]
      labels[2, point_i] = eigen_values_sorted[2]
      labels[3, point_i] = eigen_sum(eigen_values_sorted)
      labels[4, point_i] = omnivariance(eigen_values_sorted)
      labels[5, point_i] = eigenentropy(eigen_values_sorted)
      labels[6, point_i] = anisotropy(eigen_values_sorted)
      labels[7, point_i] = planarity(eigen_values_sorted)
      labels[8, point_i] = linearity(eigen_values_sorted)
      labels[9, point_i] = surface_variation(eigen_values_sorted)
      labels[10, point_i] = sphericity(eigen_values_sorted)

    return labels

if __name__ == "__main__":
  from helpers import read_roof_cloud, normalize_cloud
  file_name = "32-1-510-215-53-test-1.ply"
  cloud = read_roof_cloud(file_name)
  cloud = normalize_cloud(cloud)
  print(cloud)

  state = ScalableFeatureState(cloud)
  f = CovarianceEigenvalue(state)
  f.run_test('covariance_eigenvalue', file_name)



