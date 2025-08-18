# Information about dataset
A dataset was taken from [Kaggle](https://www.kaggle.com/datasets/mojtaba142/hotel-booking?resource=download).

# Preprocessing data
For preprocessing the data, I used the feature importance method to select the most relevant features for the model.
At the begging, I encoded categorical features using categorical encoding and got rid of the irrelevant features 
(like email, number, name and credit card) and features with too many missing values (like company and agent). 
Then I encoded the remaining categorical features using one-hot encoding.
(one-hot encoding from _pd.get_dummies_ function).


The result of the feature importance method is as follows:


# Comparing Models

To find the best model for hotels task, 

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | -        | -         | -      | -        |
| Random Forest       | -        | -         | -      | -        |
| Naive Bayes         | -        | -         | -      | -        |
| SVM                 | -        | -         | -      | -        |
