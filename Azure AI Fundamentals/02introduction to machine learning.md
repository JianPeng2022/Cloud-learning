
The metrics described above are commonly used to evaluate a regression model. In most real-world scenarios, a data scientist will use an iterative process to repeatedly train and evaluate a model, varying:
- Feature selection and preparation (choosing which features to include in the model, and calculations applied to them to help ensure a better fit).
- Algorithm selection (We explored linear regression in the previous example, but there are many other regression algorithms)
- Algorithm parameters (numeric settings to control algorithm behavior, more accurately called hyperparameters to differentiate them from the x and y parameters).
- After multiple iterations, the model that results in the best evaluation metric that's acceptable for the specific scenario is selected.

### Binary classification
Despite its name, in machine learning, **logistic regression** is used for classification, not regression. The important point is the logistic nature of the function it produces, which describes an S-shaped curve between a lower and upper value (0.0 and 1.0 **probability** when used for binary classification).

- **Accuracy**: the proportion of predictions that the model got right, (TN+TP)/(TN+FN+FP+TP). Accuracy might initially seem like a good metric to evaluate a model but consider this. Suppose 11% of the population has diabetes. You could create a model that always predicts 0, and it would achieve an accuracy of 89%, even though it makes no real attempt to differentiate between patients by evaluating their features. What we really need is a deeper understanding of how the model performs at predicting 1 for positive cases and 0 for negative cases.
- **Recall**: is a metric that measures the proportion of positive cases that the model identified correctly. TP/(TP+FN)
- **Precision**: is a similar metric to recall, but measures the proportion of predicted positive cases where the true label is actually positive. TP/(TP+FP)
- **F1-score**: is an overall metric that combines recall and precision. (2 x Precision x Recall)/(Precision + Recall)
- **Area Under the Curve (AUC)**: Another name for the recall is the true positive rate (TPR), and there's an equivalent metric called the **false positive rate (FPR)** that is calculated as FP/(FP+TN). if we were to change the **threshold** above which the model predicts true (1), it would affect the number of positive and negative predictions; and therefore change the TPR and FPR metrics. These metrics are often used to evaluate a model by plotting a received operator characteristic (ROC) curve that compares the TPR and FPR for every possible threshold value between 0.0 and 1.0. The ROC curve for a perfect model would go straight up the TPR axis on the left and then across the FPR axis at the top. Since the plot area for the curve measures 1x1, the area under this perfect curve would be 1.0 (meaning that the model is correct 100% of the time). In contrast, a diagonal line from the bottom-left to the top-right represents the results that would be achieved by randomly guessing a binary label; producing an area under the curve of 0.5. In other words, given two possible class labels, you could reasonably expect to guess correctly 50% of the time.

### Multiclass classification
One-vs-Rest (OvR) algorithms: One-vs-Rest algorithms train a binary classification function for each class, each calculating the probability that the observation is an example of the target class. Each function calculates the probability of the observation being a specific class compared to any other class. 

Multinomial algorithms: it creates a single function that returns a multi-valued output. The output is a vector (an array of values) that contains the probability distribution for all possible classes - with a probability score for each class which, when totaled, adds up to 1.0

### Clustering
There are multiple metrics that you can use to evaluate cluster separation, including:
- Average distance to cluster center: How close, on average, each point in the cluster is to the centroid of the cluster.
-Average distance to other center: How close, on average, each point in the cluster is to the centroid of all other clusters.
- Maximum distance to cluster center: The furthest distance between a point in the cluster and its centroid.
- Silhouette: A value between -1 and 1 that summarizes the ratio of distance between points in the same cluster and points in different clusters (The closer to 1, the better the cluster separation).

### deep learning

## Azure Machine Learning
Microsoft Azure Machine Learning is a cloud service for training, deploying, and managing machine learning models. 
- Exploring data and preparing it for modeling.
- Training and evaluating machine learning models.
- Registering and managing trained models.
- Deploying trained models for use by applications and services.
- Reviewing and applying responsible AI principles and practices.

Azure Machine Learning provides the following features and capabilities to support machine learning workloads:
- Centralized storage and management of datasets for model training and evaluation.
- On-demand compute resources on which you can run machine learning jobs, such as training a model.
- Automated machine learning (AutoML), which makes it easy to run multiple training jobs with different algorithms and parameters to find the best model for your data.
- Visual tools to define orchestrated pipelines for processes such as model training or inferencing.
- Integration with common machine learning frameworks such as MLflow, which makes it easier to manage model training, evaluation, and deployment at scale.
- Built-in support for visualizing and evaluating metrics for responsible AI, including model explainability, fairness assessment, and others.
