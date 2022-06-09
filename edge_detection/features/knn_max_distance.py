import numpy as np
from features.feature import ScalableFeature
from features.helpers import dist # , mean_dist

#####
#       OBS: must normalize point clouds before saving them to use this feature!
#####

class kNNMaxDistance(ScalableFeature):
  def run_at_scale(self, scale:float, knn_scale:int):
    labels = np.zeros(self.state.points.shape[0])

    for point_i, point in enumerate(self.state.points):
      [_, idx, _] = self.state.kd_tree.search_knn_vector_3d(point, knn_scale + 1)
      labels[point_i] = dist(point, self.state.points[idx[-1]])

    labels *= self.state.downsampling_factor

    return labels
    



