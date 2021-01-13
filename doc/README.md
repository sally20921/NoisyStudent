# Introduction to Self-Training 
- What if we only have enough time and money to label some of a large dataset, and choose to leave the rest unlabeled? Can this unlabeled dataset somehow be used in a classification algorithm?

- While there many flavors of semi-supervised learning, (we can train a classifier on the small amount of labeled data, and then use the classifier to make predictions on the unlabeled data. The unlabeled data predictions can be adopted as 'pseudo-labels'.) this specific technique is called self-training.

## Self-training 
1. Split the labeled data instances into train and test sets. Then, train a classification algorithm on the labeled training data.
2. Use the trained classifier to predict class labels for all of the unlabeled data instances.
3. Concatenate the 'pseudo-labeled' data with the labeled training data.
Re-train the classifier on the combined 'pseudo-labeled' and labeled training data.
4. Use the trained classifier to predict class labels for the labeled test data instances. Evaluate clasifier performance using your metric of choice.
