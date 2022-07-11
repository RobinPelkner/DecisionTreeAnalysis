# Project UniBo
Collection of various Decision Tree Algorithms to analyse data with an imbalanced class distribution with *Python 3.8*. To conquer the problem a variety of Boosting Algorithms as well as Under and Over Sampling will be tested and used.

## Table of Contents
- [Project UniBo](#project-unibo)
  - [Table of Contents](#table-of-contents)
  - [Requirements](#requirements)
  - [Introduction and Dataset](#introduction-and-dataset)
  - [First Implementations](#first-implementations)
    - [Cross Validation](#cross-validation)
  - [Boosting Algorithms](#boosting-algorithms)
  - [Oversampling](#oversampling)
  - [Undersampling](#undersampling)
  - [Results](#results)
  - [Conclusion](#conclusion)

## Requirements
Clone the Git repository

    $ git clone https://github.com/RobinPelkner/ProjectUniBo

Load the required dependecies via 

    $ cd ProjectUniBo/ 
    $ conda env create -f environment.yml 
    $ conda activate python3-env 


Most of the project is written in a Jupyter Notebook for the purpose of better manipulation of the code since it is possible to run the code piece by piece.
To run the code all at once

    $ ipython3 dtc_comp.ipynb

can be used (not recommended).    

## Introduction and Dataset
This projects aims to analyse a dataset matching various health features to people in the United States having a heart disease. The data was taken in 2020 by the Center of Disease Control and Prevention [CDC](https://www.cdc.gov/heartdisease/) and features a lot of various health indicators of the people. These people will be referred to as patients in the following. In this project a simplified version of this dataset will be used, which can be found [here](https://www.kaggle.com/datasets/kamilpytlak/personal-key-indicators-of-heart-disease).

This simplified version consists of 18 columns. Since a lot of the used algorithms can only use numerical values for analysis the data needs to be prepped. Also, the data needs to be checked for missing values and incomplete data in general. This process is done with the python script `data_prep.py`. In general, **No** is replaced with 0, while **Yes** is replaced with 1.Furthermore, the preparated data is saved for future usage in `heart_prep.py`.
Here is an overview of the feature columnds and there numerical values:
- *HeartDisease:* determines whether the patient has a heart disease or not.
- *BMI:* numerical Body Mass Index
- *Smoking*, *AlcoholDrinking* and *Stroke:* whether patients smoke/drink alcohol/had a stroke
- *PhysicalHealth* and *MentalHealth:* number of days in the last 30 days with physical/mental health issues
- *DiffWalking:* whether patients have problems walking and climbing stairs
- *Sex:* The medical sex of the patient; **Female**: 0, **Male**: 1
- *AgeCategory:* number from 0 to 12 indicate increasing age category. Youngest patients are 18
- *Race:* etnicity of the patients in numbers from 0 to 5
- *Diabetic*, *PhysicalActivity*, *Asthma* and *SkinCancer:* whether patients have one of those illnesses and whether they are physically active
- *GenHealth:* number from 0 to 4 indicating general, self evaluated health 
- *SleepTime:* average hours of sleep per 24h circle

Aim of this analysis is to find those indicators which make a heart disease most likely. In total there are 319795 valid patient entries, but only 27373 of them actually have a heart disease, which makes this a highly imbalanced class. 

<p align="center">
    <img src="figures/distributions/distribution.png" width="432"\>
</p>

Other classes with a lot of imbalance are *Stroke*, the previous illnesses and *AlcoholDrinking*, whereas *Sex*, *Race* and *AgeCategory* seem to be fairly balanced.

## First Implementations
All implementations are done in `dtc_comp.ipynb`. This project uses a variety of Decision Tree Classifiers and algorithms that depend on them. Most of the algorithms are provided by the python package [scikit-learn](https://scikit-learn.org/stable/index.html). Decision Trees are machine learning algorithms which cut the data based on the given classes. The given dataset is randomly splitted into training and test dataset in the ratio 3:2. The learning structure takes the form of a tree, where the nodes split the data based on class values into further nodes.

<p align="center"> <img src="figures/decision_tree.png" width="600"/>
</p>

The Decision Trees are built from the top to the bottom. At each further node the data will be split up so that the child nodes are as pure as possible. The so called *loss functions* are a measure for the *impurity* of the dataset and thus have to be minimised by the algorithm. How many cuts in total are made by the algorithm can be set (*max_depth*) to prevent overtraining.

To start off, the standard `DecisionTreeClassifier` (*DTC*) of **scikit-learn** is used. It provides two different criterions:
- The *Entropy* $H$ criterion describes the impurity analogous to the entropy in thermodynamics as disorder. So if there are multiple classes in a node, there is entropy here. It can be definded as  $H = - \sum_{i} p_i \text{log}_2(p_i)$, where $p_i$ is the probability of a class $i$ in a node with $N$ classes.
- The *Gini* $G$ criterion measures the frequency at which an element of the dataset will be mislabelled when it is randomly labeled. It is defined as $G = 1 - \sum_{i} p_i^2$.

To evaluate the algorithms scoring parameters will be used. In this project mostly the *accuracy* (ACC) and *recall* (REC) score will be used. They are defined by the parts of *true positive (TP)*, which is the part of data that is classifies correctly as part of the positve class, and *false positives (FP)*, which describes the data identified wrongly as part of that class. The parameters *true negative (TN)* and *false negative (FN)* are the corresponding paramters for the negative class. The accuracy is definded as $$ \text{ACC} = \frac{TP + TN}{TP + FP + TN +FN} $$ while the recall is defined as $$ \text{REC} = \frac{TP}{TN + FP} $$. Furthermore, the *Receiver operating characteristic* (ROC) curve is another used algorithm. It describes the curve of the true positive rate ($TPR = \frac{TP}{TP + FN}$) plotted against the false positive rate ($FPR = \frac{FP}{FP + TN}$). It can be evaluated with the area under the curve (ROC AuC), which is around 0.5 for a random classifier and should therefore be higher than that for a real one.

The *DTCs* have been evaluated with various maximum depth to prevent overtraining and to find the optimal values for both. 
<p align="center"> <img src="figures/max_depth/max_depth_30_Criterion_entropy_acc.png" width="400" />                                                                                                                <img src="figures/max_depth/max_depth_30_Criterion_gini_acc.png" width="400">                                                                                                                                   
  <img src="figures/max_depth/max_depth_30_Criterion_entropy_rec.png" width="400">                                                                                                     <img src="figures/max_depth/max_depth_30_Criterion_gini_rec.png" width="400" />                                                                         
</p>

As can be seen the accuracy of the test dataset and of the training data set are close to each other for both criterions and *max_depth* below 10. The accuracy is in both cases above 0.9. For *max_depth* above 10, overtraining becomes more and more visible since the curves of the accuracy test and training diverge. 
The recall score on the other hand is around 0.1 for the not overtrained area and does not move far above 0.2 at all for both criterions. This indicates a too large false positive rate, triggered by the imbalance in the distribution of Heart Diseases. All of the data is classified as "No Heart Disease". The accuracy is of course still high, since most of the data is actually classified correctly as "No Heart Disease". 
<p align="center">
    <img src="figures/distributions/distribution_dtc.png" width="400"\>
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
  
The first three algorithms are part of **scikit-learn**, while the others can be found here: [CatBoost](https://catboost.ai/), [XGBoost](https://xgboost.readthedocs.io/en/stable/).
The used algorithms are based on two different algorithms: Ada Boosting, for which a comprehensive explanation can be found [here](http://www.inf.fu-berlin.de/inst/ag-ki/adaboost4.pdf), while the other four algorithms base on Gradient Boosting. Instead of having a weight function like AdaBoost, Gradient Boosting uses Gradient descent to improve the model. The four Algorithms here are various implementations of this. *HistGradientBoosting* is based on Microsoft's [LightGBM](https://github.com/Microsoft/LightGBM), a high performance Gradient Boosting algorithm. It should also be noted that this instance has an early stop feature to prevent overtraining automatically. *CatBoost* is an alternative Gradient Boosting algorithm by Yandex. *XGBoost* uses an improved loss function instead of only using Gradient Descent, as can be seen [here](https://en.wikipedia.org/wiki/XGBoost#The_algorithm). This algorithm has only limited compatibility with **scikit-learn**.
A test was performed to find an optimal value for the number of iterations ("estimators"). The plots for one algorithm are shown as an example since all algorithms provide similar scores. It can be seen, that the accuracy value is stable even for high numbers of estimators, while 
the recall value has improved in comparison to simple DTCs, but is still far away from an acceptable value. The plots shown were made using AdaBoost, the other algorithms produce very similar results.
<p align="center"> <img src="figures/boosting_est/accAdaBoost.png" width="400" />                                                                                                                <img src="figures/boosting_est/recAdaBoost.png" width="400">        
<p/>


For all included algorithms cross validation is performed for well performing numbers of estimators/depths of trees. The results are included in the table below. It becomes clear that the Boosting algorithms do not improve the overall score by much. All of these recall values are not acceptable for a further analysis.
|           |DTC Entropy|DTC Gini|AdaBoost|GradientBoost|HistGradientB.|CatBoost|XGBoost| 
|---        |---|---|---|---|---|---|---|
|Accuracy   |0.9134 ± 0.0006|0.9123 ± 0.0006|0.9148 ± 0.0012|0.9160 ± 0.0004|0.9157 ± 0.0006|0.9152 ± 0.0007|0.9147 ± 0.0005|
|ROC AuC    |0.8130 ± 0.0035|0.8132 ± 0.0032|0.8377 ± 0.0026|0.8441 ± 0.0020|0.8438 ± 0.0017|0.8430 ± 0.0019|0.8386 ± 0.0019|
|Recall     |0.0768 ± 0.0144|0.0868 ± 0.0148|0.1028 ± 0.0062|0.0941 ± 0.0040|0.0775 ± 0.0067| 0.0865 ± 0.0058|0.0998 ± 0.0029|

## Oversampling
One method of improving faulty results due to imbalanced classes is the so called Oversampling. It includes copies of the minority class to the dataset. This is a method usually used when there is not a lot of data available since it makes the training more resource costly. Inclusion of more underrepresented data in combination with boosting algorithms should improve the recall score by a lot. The package [Imbalanced-learn](https://imbalanced-learn.org/stable/index.html) includes many Oversampling algorithms, three of them will be used in this analysis: The *RandomOverSampler* copies samples at random from the underrepresented class. Apart from that there are two other popular over sampling techniques: *Synthetic Minority Oversampling Technique* (SMOTE) and the *Adaptive Synthetic Sampling Approach* (ADASYN). SMOTE creates synthetic data by applying a nearest neigbour algorithm to the original data thus not copying already existing data, but creating similar samples. ADASYN follows a similar approach where synthetic data is generated which is harder to learn for the models thus improving the learning procedure as well; details on this can be found [here](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4633969). The scoring results can be found in the plots below.

<p align="center"> <img src="figures/Sampling/AccuracyOver.png" width="500"/>
</p>
<p align="center"> <img src="figures/Sampling/RecallOver.png" width="500"/>
</p>
<p align="center"> <img src="figures/Sampling/ROCAuCOver.png" width="500"/>
</p>

It becomes apparent that the accuracy has been lowered in general in comparison to the models without sampling especially when using the *RandomOverSampler*. The *RandomOverSampler*  on the other hand provides the best recall values of almost 0.8 for all models, while the not-boosted algorithms seem to work best with the SMOTE and ADASYN in this regard. The ROC AuC values are very similar to the values found before.

All in all it can be said, that *Over Sampling* improved the quality of the recall value significantly at the slight cost of decreased accuracy.  

## Undersampling
Another often used method to improve datasets with imbalanced classes is Undersampling. Undersampling is basically the opposite of Oversampling. Instead of generating more data for the minority class, the majority class gets reduced. This project takes a look at three Undersampling procedures of the [Imbalanced-learn](https://imbalanced-learn.org/stable/index.html) package. There is the *RandomUnderSampler* which removes samples randomly, the *NearMiss* (NM), which adds heuristic selection rules based on a kNN algorithm, and *InstanceHardnessThreshold* (IHT), which removes samples based on their probability of being classified wrong. The results of the scoring tests are shown below.

<p align="center"> <img src="figures/Sampling/AccuracyUnder.png" width="500"/>
</p>
<p align="center"> <img src="figures/Sampling/RecallUnder.png" width="500"/>
</p>
<p align="center"> <img src="figures/Sampling/ROCAuCUnder.png" width="500"/>
</p>

In comparison to the Oversampling models, the accuracy decreased again. For *RandomUnderSampler* and IHT the accuracy is now around 0.7, while NM has a significantly stronger drecrease. The recall value is now for all samplers and all boosting algorithms above 0.8, where NM achieves the highest score. The ROC AuC score is similar to before with the exception of the NM algorithm.

Under Sampling algorithms seem to be working as well. With the trade-off of less accuracy even more recall could be gained in comparison to Over Sampling. The only exception seems to be the Near Miss algorithm which loses a lot of accuracy over the others. 

## Results
To gain the final results, the IHT algorithm with *HistGradientBoost* is used, since its recall value is the best, while also having reasonable accuracy. For comparative reasons, other IHT models with AdaBoost and CatBoost will be used. 

To evaluate the importance, *permutation importance* will be used. It is designed to evaluate the decrease of a scoring metric, when features are shuffeled randomly. The scoring metric used here is *ROC AuC*. 

The feature importances of all models are shown below in the form of box plots.

<p align="center"> <img src="figures/importance/ada.png" width="500"/>
</p>
<p align="center"> <img src="figures/importance/cat.png" width="500"/>
</p>
<p align="center"> <img src="figures/importance/hgb.png" width="500"/>
</p>

All models bring similar results, although the order of features are not completely the same. All models deem the age of the patient to be the most important feature, followed by general health, the sex whether the patient had a stroke before and whether they have diabetes. Other features make almost no impact at all like mental health, BMI and physical activity. 

This result is not unexpected since it is known that the probability of heart diseases increases with age, certain sexes being more subsceptible to certain illnesses and other health issues not necessarily influencing the heart. Other features, like General Health, are self evaluated and thus not completely quantifiable measures. This *human factor* can have enourmus effects on the final result.

## Conclusion 
In this project it was shown that imbalanced class distributions have significant impact on the viability of decision tree and boosted decision tree algortihms. Under given circumstances (like here) the algorithms can behave completely unreliable. Over and Under Sampling were introduced as possible solutions. They increased the recall values by a huge amount at a slight cost of accuracy. With the help of undersampling algorithms and boosted decision trees the feature importance was analysed. This of course is not a global result which can applied to every dataset; the outcome is always depending on the used dataset.