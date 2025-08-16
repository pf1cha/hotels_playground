import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(confusion_matrix, labels):
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, linewidths=.5)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()


def correlation_heatmap(df):
    numerics_columns = df.select_dtypes(include=['int64', 'float64']).columns
    new_df = df[numerics_columns]
    plt.figure(figsize=(20, 10))
    correlation_matrix = new_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap')
    plt.show()
