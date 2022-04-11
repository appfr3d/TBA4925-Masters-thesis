import os
import numpy as np
import pandas as pd
import open3d as o3d
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

np.set_printoptions(precision=4)

def get_dataset_path(file_name):
  current_folder = os.path.dirname(os.path.abspath(__file__))
  project_folder = os.path.dirname(current_folder)
  return os.path.join(project_folder, "data/point_clouds/classified_roofs/calculated_values", file_name)

training_data_file_names = ["32-1-510-215-53-test-1.csv", "32-1-510-215-53-test-2.csv", "32-1-510-215-53-test-3.csv"] 
evaluation_data_file_names = ["32-1-510-215-53-test-4.csv", "32-1-510-215-53-test-5.csv"]

def calculate_weight(targets):
  one_targets = sum([1 if x == 1 else 0 for x in targets])
  zero_targets = sum([1 if x == 0 else 0 for x in targets])
  one_weight = zero_targets/(one_targets+zero_targets)
  zero_weight = one_targets/(one_targets+zero_targets)
  return [one_weight if target == 1 else zero_weight for target in targets]

def main():
  training_data = pd.DataFrame()
  for file_name in training_data_file_names:
    data = pd.read_csv(get_dataset_path(file_name), index_col=0)
    training_data = pd.concat([training_data, data])

  # Split in traning and testing splits
  features = training_data.drop("target", axis=1)
  targets = training_data["target"]
  X_train, X_validation, y_train, y_validation = train_test_split(features, targets, train_size=0.8, random_state=1234)
  
  # Weight the edge /non-edge classes
  target_weight = calculate_weight(y_train)

  # TODO: test with categorical?
  # categorical_features = []

  train_pool = Pool(
    data=X_train,
    label=y_train,
    weight=target_weight,
  )

  model = CatBoostClassifier(
    iterations=50,
    learning_rate=0.05,
    random_seed=42,
  )

  # model = CatBoostClassifier(
  #   iterations=700,
  #   # use_best_model=True,
  #   eval_metric="AUC",
  #   od_type="Iter",
  #   od_wait=40,

  #   learning_rate=0.07650534937760373, 
  #   depth=4,
  #   l2_leaf_reg=4.4385423625938545,
  #   one_hot_max_size=3,
  #   border_count=254,

  #   random_seed=42,
  #   loss_function="RMSE",
  #   verbose=True,

  #   eval_set
  # )

  model.fit(train_pool, eval_set=(X_validation, y_validation), verbose=True)

  # Calculate feature importance
  feature_importance_list = model.get_feature_importance(type="FeatureImportance")
  feature_list = features.columns.tolist()
  feature_importance = pd.DataFrame({"feature": feature_list, "importance": feature_importance_list})
  feature_importance.sort_values(by=["importance"], ascending=[False], inplace=True)
  print("\nFeature importance:")
  print(feature_importance)

  evaluation_data = pd.DataFrame()
  for file_name in evaluation_data_file_names:
    data = pd.read_csv(get_dataset_path(file_name), index_col=0)
    evaluation_data = pd.concat([evaluation_data, data])

  evaluation_features = evaluation_data.drop("target", axis=1)
  evaluation_targets = evaluation_data["target"].to_numpy()

  predictions = model.predict(evaluation_features)

  # Evaluation metrics
  print("Evaluation:")

  point_percent = np.sum(np.equal(predictions, evaluation_targets))/predictions.shape[0]
  print("\tPoint perecentage:", point_percent)

  # positives = np.sum(predictions)
  # negatives = predictions.shape[0] - positives
  # true_positives = np.bitwise_and(predictions, evaluation_targets)

  precision = precision_score(evaluation_targets, predictions)
  recall = recall_score(evaluation_targets, predictions)
  print("\tPrecision        :", precision)
  print("\tRecall           :", recall)

  # Visualize result
  for file_name in evaluation_data_file_names:
    visualization_data = pd.read_csv(get_dataset_path(file_name), index_col=0)
    visualization_features = visualization_data.drop("target", axis=1)

    predictions = model.predict(visualization_features)
    pcd = o3d.geometry.PointCloud()
    points = visualization_data[["x","y","z"]].to_numpy()
    pcd.points = o3d.utility.Vector3dVector(points)

    colors = np.zeros((predictions.shape[0], 3))
    colors += [0.6, 0.6, 0.6]
    colors[predictions < 0.5] = [0, 1, 0] # Color positive values as green

    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
  main()
