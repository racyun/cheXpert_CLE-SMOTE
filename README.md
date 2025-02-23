Among the many intriguing problems in AI, one question stands out for me: how can we address class imbalance in datasets to improve model robustness? Medical datasets often face not only inter-class imbalance (uneven distribution across classes) but also intra-class imbalance (uneven distribution within subclasses). This presents a fundamental challenge for ML models in detecting rare diseases or anomalies.
To explore this question, I undertook a research project addressing a critical challenge in healthcare—the backlog of unclassified chest X-rays—by improving ML model performance in the face of limited and imbalanced data. Together with researchers Cara Wang and Zaid Nabulsi, I developed Contrastive Learning-Enhanced SMOTE (CLE-SMOTE), a novel data augmentation method. Unlike traditional oversampling techniques that can generate noisy synthetic examples, CLE-SMOTE combines supervised contrastive learning and a capping mechanism to mitigate both inter- and intra-class imbalance. By using contrastive learning to differentiate between majority and minority classes and applying cosine distance to filter noisy synthetic data, we improved model performance. Tested on the CheXpert dataset, CLE-SMOTE achieved state-of-the-art results, offering a scalable solution for AI-powered chest X-ray analysis. We are currently submitting this work for publication.
