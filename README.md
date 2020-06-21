## Noisy student

2019_2 artificial intelligence application term project

## Introduction

On November 11, the Google Research Brain Team broke the fix for the train-test resolution discrepancy, SOTA of ImageNet, and set a new record with a gap of 1% in the TOP-1 error. With SOTA changing often, breaking new records in ImageNet did not seem significant. However, unlike the pretraining with labeled data, the first place with only unlabeled data and ImageNet data is a meaningful result. The idea of ​​the training method is also interesting because it is simple and inspired by the actual teacher-student relationship.

## Method

Noisy student is based on the teacher-student relationship seen in reality. The steps can be divided into four steps.

    Learn the Teacher model with labeled data.
    Predict the labeling of unlabeled data with the Teacher model. (pseudo labels)
    Learn about labeled and unlabeled data with the Student model.
    After completing the training, the student model becomes the teacher model and repeats 2 and 3 with the new student model.

First, learn the model that will be the teacher model with labeled data. At this point, you get a big regularization effect by giving your model noisy. At the end of the lesson, the teacher model predicts the logit of unlabeled data. At this time, noisy is used for high accuracy. The generated logit is called pseudo label, and it can be used as soft before argmax or hard as argmax. The student model is trained using labeled data, unlabeled data + pseudo labels. In this case, the weighted data is labeled as unlabeled data and unlabeled data. This learned student becomes a teacher and repeats the above. There are several conditions for this method to work. The first is noisy. Noisy students give noisy when learning models using Dropout, stochastic depth, and RandAugmentation. They have regularization effects and robustly learn about other inputs. The second is whether the student model can accommodate more data. This should be large enough to accommodate the performance of the teacher model while at the same time getting higher performance than the teacher model from the labeled data. Recent studies have used distillation of models that have already been trained using teacher-student relationships. They use a smaller model than the teacher model, while the noisy student uses a larger model because it aims to achieve better performance than the teacher model.

##Experiment

I used cifar10 with 32x32 resolution for quick experiments. This dataset consists of 50000 training data and 10000 test data. This is divided into 5000 labeled data and 45000 unlabeled data. This distribution is divided by 9 times considering that 14M of ImageNe used as labeled data used in the Noisy student paper and 130M of Google internal data used as unlabeled data are used. When I first planned the experiment, I tried to use Resnet or Vgg. However, these models had only 5,000 labeded data, so the problem was that the accuracy was lowered as the model size increased. Therefore, we chose four models by stacking a few layers and training several models. They learned that with 5000 data, the bigger the model, the better the accuracy would be by 1-2%. Convnet5, Convnet6 and Resnet9 with skip connectiond consisting of several layers of convolution layers and feedforwad on the first layer. The smallest Convnet5 has 100,000 parameters and the largest Resnet9 has 300,000 parameters, all of which are very small compared to the models used today.

Rand Augment, used as a noisy for learning, has two parameters. One determines which of the 14 augment methods to choose, and the other determines the upper bound of the augment strength (1 to 10). Compared to other auto augmentation, the search space (14x10) is very small, but I had to search for parameters several times. Grid search found the best parameters for 5000 data. As the size of the model changes or the data changes, the appropriate Rand Augment parameter may change, but the same parameters were used for all experiments in terms of time and resources.

## Result
<img src="assets/result1.png" width="500">
<img src="assets/result2.png" width="500">

First we found the parameters of RandAugmentation as grid-serach. Due to lack of time and resources, we only ran 5000 data points for the smallest Convnet5 and the largest Resnet9. As a result, the two models, Convnet5 and Resnet9, performed best when the two RandAugmentation parameters were (5, 10). Therefore, in all subsequent trainings, we applied (5, 10) regardless of the amount and model of data.

After finding the parameters of RandAugmentation, we trained four models without using noisy-student. Although noisy-student was not used, the dropout and auto-augmentation were used to improve the performance by 6-7% compared to when it was not. RandAugmentation uses the same (5, 10) found above. Convnet5 had an accuracy of 63.6%, and as the model grew, there was an increase in accuracy of 1-2%. Using this as a baseline, we observed a performance improvement when using noisy-student.

The smallest model, Convnet5, only learned about labeled data. After that, we studied Convnet6 as Convnet5, Consnet6 as Convnet6 and Resnet9 as Resnet6 as the noisy-student method. As a result of learning, using noisy-student improved 4 ~ 4.8% accuracy than unused.

In the original paper, if the teacher model had high accuracy, there was no big difference between using soft pseudo label or hard pseudo label. However, if the accuracy is low, using a soft pseudo label shows higher instincts. As a result of the experiment, there was little difference in accuracy between using soft pseudo label and using hard pseudo label.

Compared to the performance of the labeled data size, the noisy student has higher accuracy than using 8000 labeled data, but lower performance than using 9000.

## Execution

    Install python package if you already have Python and have gpu and cuda installed that support cuda

pip install -r requirements.txt

    Run data_split.ipynb in jupyter to download and split data

    run.ipynbroll experiment

## Code

We do four things in the run.ipynb file.

    1. Search RandAugment's parameters

    2. Baseline performance of four models

    3. Noisy-Student Execution

    4. Accurac, the largest model, Resnet9
