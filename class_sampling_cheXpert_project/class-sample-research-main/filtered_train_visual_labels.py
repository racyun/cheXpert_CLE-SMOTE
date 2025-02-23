import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def create_correlation_map(df, columns, title):
    subset_df = df[columns]
    f, ax = plt.subplots(figsize=(10, 8))
    corr = subset_df.corr()
    
    plt.title(title)
    sns.heatmap(corr,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmin=-1.0, vmax=1.0,
                square=True, ax=ax)
    plt.show()


def create_filtered_labels_df():
  train_visualCheXbert_labels = pd.read_csv('../../chexlocalize/CheXpert/train_visualCheXbert.csv')

  columns_list = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices", "No Finding"]
  for item in columns_list:
    column = train_visualCheXbert_labels.loc[:,item]


  # columns_list = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "No Finding"]
  # create_correlation_map(train_visualCheXbert_labels, columns_list, "Correlation Map")

  def focal_multifocal_lung_opacity(row):
    if row["Atelectasis"]==1 or row["Lung Opacity"]==1 or row["Consolidation"]==1 or row["Pneumonia"]==1:
      return 1
    return 0

  def enlarged_cardiac_silhouette(row):
    if row["Cardiomegaly"]==1 or row["Enlarged Cardiomediastinum"]==1:
      return 1
    return 0


  train_visualCheXbert_labels["focal_multifocal_lung_opacity"] = train_visualCheXbert_labels.apply(focal_multifocal_lung_opacity, axis=1)
  train_visualCheXbert_labels["enlarged_cardiac_silhouette"] = train_visualCheXbert_labels.apply(enlarged_cardiac_silhouette, axis=1)
  filtered_train_visualCheXbert_labels = train_visualCheXbert_labels.drop(columns=['Atelectasis', 'Lung Opacity', 'Consolidation', 'Pneumonia', 'Enlarged Cardiomediastinum', 'Cardiomegaly'])


  NEW_CONDITION_NAMES = ['Lung Lesion', 'Edema', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'focal_multifocal_lung_opacity', 'enlarged_cardiac_silhouette']
  def row_has_condition(row):
    for condition in NEW_CONDITION_NAMES:
      if row[condition] == 1.0:
        if condition == 'Lung Lesion':
          return 1
        if condition == 'Edema':
          return 2
        if condition == 'Pneumothorax':
          return 3
        if condition == 'Pleural Effusion':
          return 4
        if condition == 'Pleural Other':
          return 5
        if condition == 'Fracture':
          return 6
        if condition == 'focal_multifocal_lung_opacity':
          return 7
        if condition == 'enlarged_cardiac_silhouette':
          return 8
    return 0

  filtered_train_visualCheXbert_labels["Condition"] = train_visualCheXbert_labels.apply(row_has_condition, axis=1)

  filtered_train_visualCheXbert_labels["has_finding_no_condition"] = filtered_train_visualCheXbert_labels.apply(lambda row: row["No Finding"]==0 and row["Condition"]==0, axis=1)
  filtered_train_visualCheXbert_labels["no_finding_has_condition"] = filtered_train_visualCheXbert_labels.apply(lambda row: row["No Finding"]==1 and row["Condition"]==1, axis=1)

  #Filter labels down the rows where no_finding_has_condition is true and where has_finding_no_condition is true
  no_finding_has_condition_rows = filtered_train_visualCheXbert_labels.loc[filtered_train_visualCheXbert_labels['no_finding_has_condition']]
  has_finding_no_condition_rows = filtered_train_visualCheXbert_labels.loc[filtered_train_visualCheXbert_labels['has_finding_no_condition']]
  filtered_train_visualCheXbert_labels = filtered_train_visualCheXbert_labels.loc[~ filtered_train_visualCheXbert_labels['has_finding_no_condition']]
  filtered_train_visualCheXbert_labels = filtered_train_visualCheXbert_labels.loc[~ filtered_train_visualCheXbert_labels['no_finding_has_condition']]

  def num_conditions(row):
    condition_count = 0
    for condition in NEW_CONDITION_NAMES:
      if row[condition] == 1.0:
        condition_count += 1
    return condition_count

  filtered_train_visualCheXbert_labels["num_conditions"] = filtered_train_visualCheXbert_labels.apply(num_conditions, axis=1)
  filtered_train_visualCheXbert_labels = filtered_train_visualCheXbert_labels.loc[filtered_train_visualCheXbert_labels["num_conditions"]<=1]
  return filtered_train_visualCheXbert_labels
