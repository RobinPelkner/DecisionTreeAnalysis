# Project UniBo
Collection of various Decision Tree Algorithms to analyze data with an unbalanced class distribution with *Python 3.8*. To conquer the problem a variety of Boosting Algorithms as well as Under and Over Sampling will be tested.

## Table of Contents
- [Requirements](#requirements)
- [Introduction and Dataset](#introduction-and-dataset)
- [First Implementation](#first-implementations)
    - [Cross Validation](#cross-validation)
- [Boosting Algorithms](#boosting-algorithms)

## Requirements
Clone the Git repository
    $ git clone https://github.com/RobinPelkner/ProjectUniBo

Load the required dependecies via 
    $ cd ProjectUniBo
    $ conda env create -f environment.yml
    $ conda activate python3-env

Most of the project is written in a Jupyter Notebook for the purpose of better manipulation of the code since it is possible to run the code piece by piece.

## Introduction and Dataset
This projects aims to analyse a dataset matching various health features to the patients having a heart disease. The data was taken in 2020 by the Center of Disease Control and Prevention [CDC](https://www.cdc.gov/heartdisease/) and features a lot of various health indicators of the patients. In this project a simplified version of this dataset will be used, which can be found [here](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease).

This simplified version consists of 18 columns. Since a lot of the used algorithms can only use numerical values for analysis the data needs to be prepped. Also, the data needs to be checked for missing values and incomplete data in general. This process is done with the python script `data_prep.py`. In general, **No** is replaced with 0, while **Yes** is replaced with 1.Furthermore, the preparated data is saved for future usage in `heart_prep.py`.
Here is an overview of the feature columnds and there numerical values:
- *HeartDisease*: Determines wether the patient has a heart disease or not.
- *BMI*: numerical Body Mass Index
- *Smoking*, *AlcoholDrinking* and *Stroke*: Wether patients smoke/drink alcohol/had a stroke
- *PhysicalHealth* and *MentalHealth*: Number of days in the last 30 days with physical/mental health issues
- *DiffWalking*: Wether patients have problems walking and climbing stairs
- *Sex*: The medical sex of the patient; **Female**: 0, **Male**: 1
- *AgeCategory*: number from 0 to 12 indicate increasing age category. Youngest patients are 18
- *Race*: etnicity of the patients in numbers from 0 to 5
- *Diabetic*, *PhysicalActivity*, *Asthma* and *SkinCancer*: wether patients have one of those illnesses and wether they are physically active
- *GenHealth*: number from 0 to 4 indicating general, self evaluated health 
- *SleepTime*: average hours of sleep per 24h circle

Aim of this analysis is to find those indicators which make a heart disease most likely. In total there are 319795 valid patient entries, but only 27373 of them actually have a heart disease, which makes this a highly unbalanced class. 

<p align="center">
    <img src="./figures/distribution.png" width="432"\>
</p>

Other classes with a lot of imbalance are *Stroke*, the previous illnesses and *AlcoholDrinking*,whereas *Sex*, *Race* and *AgeCategory* seem to be fairly balanced.

## First Implementations
All implementations are done in `dtc_comp.ipynb`. This project uses a variety of Decision Tree Classifier and algorithms that depend on them. Most of the algorithms are provided by the python package **[scikit-learn]**(https://scikit-learn.org/stable/index.html). Decision Trees are machine learning algorithms which cut the data based on the given classes. The given dataset is randomly splitted into training and test dataset in the ratio 3:2. The learning structure takes the form of a tree, where the class labels are leaves and the actual values are the branches.
<p align="center">
    <img src="./figures/decision_tree.png" width="550"\>
</p>
The Decision Trees are build from the top to the bottom. At each further node the data will be split up that the child nodes are as pure as possible. The so called *loss functions* are a measure for the *impurity* of the dataset and thus has to be minimized by the algorithm. How many cuts in total are made by the algorithm can be set (*max_depth*) to prevent overtraining.

To start off, the standard `DecisionTreeClassifier` (*DTC*) of **scikit-learn** is used. It provides two different criterions:
- The *Entropy* $H$ criterion describes the impurity analogous to the entropy in thermodynamics as disorder. So if there are multiple classes in a node, there is entropy here. It can be definded as  $H = - \sum_{i=1}^{N} p_i \text(log)_2(p_i)$, where $p_i$ is the probability of a class in a node with $N$ classes.
- The *Gini* $G$ criterion measures the frequency at which an element of the dataset will be mislabelled when it is randomly labeled. It is defined as $G = 1 - \sum_{i}^{N} p_i^2$.

To evaluate the algorithms scoring parameters will be used. In this project mostly the *accuracy* (ACC) and *recall* (REC) score will be used. They are defined by the parts of *true positive (TP)*, *false positives (FP)*, *true negative (TN)* and *false negative (FN)* results.The accuracy is definded as $$ \text{ACC} = \frac{TP + TN}{TP + FP + TN +FN} $$ whilethe recall is defined as $$ \text{REC} = \frac{TP}{TN + FP} $$. Furthermore, the *Receiver operating characteristic* (ROC) curve is another used algorithm. It describes the curve of the true positive rate ($TPR = \frac{TP}{TP + FN}$) plotted against the false positive rate ($FPR = \frac{FP}{FP + TN}$). It can be evaluated with the area under the curve (ROC AuC), which is around 0.5 for a random classifier and should therefore be higher than that for a real one.

The *DTCs* have been evaluated with various maximum depth to prevent overtraining and to find the optimal values for both. 
|<p align="center"> <img src="./figures/max_depth_30_Criterion_entropy_acc.png" width="400" />                                                                                                     |           <img src="./figures/max_depth_30_Criterion_gini_acc.png" width="400">  /                                                                                                                                 |
|  <img src="./figures/max_depth_30_Criterion_entropy_rec.png" width="400"> /                                                                                                     |<img src="./figures/max_depth_30_Criterion_gini_rec.png" width="400" />                                                                         |
</p>
As can be seen the accuracy of the test dataset and of the training data set are close to each other for both criterions and `max_depth` below 10. The accuracy is in both cases above 0.9. For `max_depth` above 10, overtraining becomes more and more visible since the curves of the accuracy of the test and the training dataset move away from each other. 
The recall score on the other hand is around 0.1 for the not overtrained area and does not move far above 0.2 at all for both criterions. This indicates a too large false positive rate, triggered by the imbalance in the distribution of Heart Diseases. All of the data is classified as "No Heart Disease". The accuracy is of course still high, since most of the data is actually classified correctly as "No Heart Disease". 
<p align="center">
    <img src="./figures/distribution_dtc.png" width="400"\>
</p>
Aim of this project is to circumvent this problem in various ways and to find an actually working procedure.

### Cross Validation
Cross Validation is a method of validating the used models. The training dataset gets divided into $k$ subsets. The model gets tested on all $k$ subsets individually after being trained 
on the remaining $k-1$ subsets. This makes the model less susceptible to noise in the data 
and gives a more accurate scoring. In this project the parameter is set to $k=5$ and the evaluation method of **scikit-learn** will be used.

## Boosting Algorithms
To improve the standard DTC one can use *Boosting* to improve the outcome. Boosting algorithms are ensemble methods. They take simple algorithms (*weak learner*) like DTC, iterate them sequentialy based on the weighted iterations before. In the end they return a weighted average  making them a *strong learner*, thus reducing noise, variance and bias. The weighting process itself is dependend on the algorithm. This project will take a look at various Boosting algorithms:
- AdaBoost
- GradientBoosting
- HistGradientBoosting
- CatBoost
- eXtremeGradientBoosting
The first three algorithms are part of **scikit-learn**, while the others are standalones, which can be found here: [CatBoost](https://catboost.ai/), [XGBoost](https://xgboost.readthedocs.io/en/stable/).
