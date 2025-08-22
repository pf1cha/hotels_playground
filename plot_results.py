import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(confusion_matrix, labels, name=None):
    plt.figure(figsize=(10, 7))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, linewidths=.5)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'figures/{name}')


def correlation_heatmap(df):
    numerics_columns = df.select_dtypes(include=['int64', 'float64']).columns
    new_df = df[numerics_columns]
    plt.figure(figsize=(40, 20))
    correlation_matrix = new_df.corr()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title('Correlation Heatmap')
    plt.savefig('figures/correlation_heatmap.png')


def plot_feature_importance(feature_importance_df):
    plt.figure(figsize=(20, 10))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importance')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.savefig('figures/feature_importance.png')