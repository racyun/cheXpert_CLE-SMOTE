import pandas as pd
from torch.utils.data import Dataset
import os
from PIL import Image
import tqdm
import numpy as np
import filtered_train_visual_labels


def map_img_to_labels(patient_id, test_labels):
  row_wanted = test_labels[test_labels['Path'].str.contains(patient_id)]
  row_idx = list(row_wanted.index)[0]
  labels_dict = {}
  labels = test_labels.loc[row_idx]

  values_labels_list = labels.values.tolist()
  for index, item in enumerate(values_labels_list):
    labels_dict[labels.index[index]] = item

  print(labels_dict)


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        #self.img_labels = pd.read_csv('../../chexlocalize/CheXpert/test_labels.csv')
        self.img_labels = filtered_train_visual_labels.create_filtered_labels_df()

        self.img_labels = self.img_labels[self.img_labels['Path'].str.contains("frontal")]
        self.targets = self.img_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        self.imgs = []
        self.patient_ids = []

        img_dir_list = os.listdir(self.img_dir)
        for patient_id in tqdm.tqdm(img_dir_list):
          if patient_id==".DS_Store":
            continue
          img_path = os.path.join(img_dir, patient_id, 'study1')
          img_list = os.listdir(img_path)
          filename = None
          for item in img_list:
            if "frontal" in item:
              filename = item
              break
          if filename:
            img_path = os.path.join(img_path, filename)
          else:
            continue
          img = Image.open(img_path)
          img = np.asarray(img)
          self.imgs.append(np.asarray(img))
          self.patient_ids.append(patient_id)

        #imgs = torch.tensor(imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        #label = self.img_labels.iloc[idx, 1].to_dict()
        label = self.img_labels.iloc[idx][['No Finding', 'Lung Lesion', 'Edema', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'focal_multifocal_lung_opacity', 'enlarged_cardiac_silhouette']]
        label = self.img_labels.iloc[idx]['No Finding']
        label = label.tolist()
        if self.transform:
            image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return image, label

