import numpy as np
import laspy
data = np.loadtxt('data/10447852.txt').astype(np.float64)


xyz = data[:, 0:3]
sem_labels = data[:, -1].astype(np.int32)
ins_labels = data[:, -2].astype(np.int32)

k = 40
l = 1
r = 0.09

def dist(p1, p2):
  return np.sqrt(np.abs(p1[0] - p2[0])**2 + np.abs(p1[1] - p2[1])**2 + np.abs(p1[2] - p2[2])**2)

def manhattan_dist(p1, p2):
  return np.abs(p1[0] - p2[0]) + np.abs(p1[1] - p2[1]) + np.abs(p1[2] - p2[2])

def mean_dist(p, pts):
  all_dist = np.array([dist(p, x) for x in pts])
  return np.mean(all_dist)

def knn_naive(points, query_point_index):
  # Find k nearest neighbors of point
  neighbor_dist = []
  for i, n in enumerate(points):
    if i != query_point_index:
      neighbor_dist.append((dist(point, n), i))
  
  sorted_indices = np.array(sorted(neighbor_dist, key=lambda p: p[0]))[:,1].astype(np.int32)

  return np.array(xyz[sorted_indices[:k]])


def ball_query_naive(points, query_point_index):
  # Find nearest neighbors within radius r
  neighbor_dist = []
  for i, n in enumerate(points):
    if i != query_point_index:
      neighbor_dist.append((dist(point, n), i))
  
  # sorted_indices = np.array(sorted(neighbor_dist, key=lambda p: p[0]))[:,1].astype(np.int32)
  distance = np.array(neighbor_dist)

  distance_indices = distance[distance[:,0] < r][:, 1].astype(np.int32)

  return xyz[distance_indices]

def normalize(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


values = np.zeros(xyz.shape[0])

for i_point, point in enumerate(xyz):
  # Find k nearest neighbors of point
  nearest_neighbors = knn_naive(xyz, i_point)
  closest_neighbor = nearest_neighbors[0]

  # Find nearest neighbors with ball query
  # nearest_neighbors = ball_query_naive(xyz, i_point)
  # closest_neighbor = nearest_neighbors[np.argmin(nearest_neighbors[:,0])]

  centroid = (1/k) * np.sum(nearest_neighbors, axis=0)

  # min_dist = dist(point, closest_neighbor)
  min_dist = mean_dist(point, nearest_neighbors[:10])

  values[i_point] = dist(centroid, point) - l*min_dist


# Normalize values between 0 and 1
# print('before', values.max(), values.min())
norm = normalize(values)
print('first', norm.max(), norm.min())

# Remove outliers
print('median', np.median(norm))
print('mean', np.mean(norm))
cap = np.median(norm)*2
print('cap', cap)
print('num over  cap', np.sum(norm > cap))
print('num under cap', np.sum(norm <= cap))
norm = np.clip(norm, 0, cap)
norm = normalize(norm)

# all_edges = np.array(edges).astype(np.float32)

# Store as .las file
header = laspy.LasHeader(version='1.4', point_format=7)
las = laspy.LasData(header)
las.x = xyz[:, 0]
las.y = xyz[:, 1]
las.z = xyz[:, 2]
# las.red = rgb[:, 0]
# las.green = rgb[:, 1]
# las.blue = rgb[:, 2]
las.intensity = norm
# las.classification = seg_values

# Store the file
las.write('results/manhattan-' + str(l) + '-' + str(k) + '.las')

