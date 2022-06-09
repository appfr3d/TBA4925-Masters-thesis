import numpy as np
from features.feature import Feature
import open3d as o3d
from matplotlib import cm
import copy

#####
#       OBS: must normalize point clouds before saving them to use this feature!
#####

class z_pow2(Feature):
  def run(self, verbose=False):
    Z = self.state.points[:, 2]
    # diff = np.max(Z) + np.min(Z)
    # Z -= diff/2
    return np.power(Z, 2)