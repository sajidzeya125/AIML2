import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv('Titanic-Dataset.csv')

## 1. Generate summary statistics
print("=== Summary Statistics ===")
print(df.describe())

# Additional statistics for categorical data
print("\n=== Categorical Data Summary ===")
print(df.describe(include=['O']))

## 2. Create histograms and boxplots for numeric features
numeric_cols = ['Age', 'Fare', 'SibSp', 'Parch']
plt.figure(figsize=(15, 10))

# Histograms
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(2, 4, i)
    sns.histplot(df[col], kde=True)
    plt.title(f'Histogram of {col}')

# Boxplots
for i, col in enumerate(numeric_cols, 5):
    plt.subplot(2, 4, i)
    sns.boxplot(y=df[col])
    plt.title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()

## 3. Use pairplot/correlation matrix for feature relationships
# Pairplot for numeric features
print("\n=== Pairplot of Numeric Features ===")
sns.pairplot(df[numeric_cols + ['Survived']], hue='Survived')
plt.show()

# Correlation matrix
print("\n=== Correlation Matrix ===")
corr_matrix = df[numeric_cols + ['Survived']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

## 4. Identify patterns, trends, or anomalies
# Survival rate by class
print("\n=== Survival Rate by Passenger Class ===")
survival_by_class = df.groupby('Pclass')['Survived'].mean()
print(survival_by_class)
survival_by_class.plot(kind='bar')
plt.title('Survival Rate by Passenger Class')
plt.ylabel('Survival Rate')
plt.show()

# Survival rate by gender
print("\n=== Survival Rate by Gender ===")
survival_by_gender = df.groupby('Sex')['Survived'].mean()
print(survival_by_gender)
survival_by_gender.plot(kind='bar')
plt.title('Survival Rate by Gender')
plt.ylabel('Survival Rate')
plt.show()

# Age distribution by survival
print("\n=== Age Distribution by Survival ===")
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='Age', hue='Survived', kde=True, bins=30)
plt.title('Age Distribution by Survival Status')
plt.show()

## 5. Make basic feature-level inferences
print("\n=== Key Inferences ===")
print("1. Higher passenger classes (1st class) had better survival rates")
print("2. Females had significantly higher survival rates than males")
print("3. Children (especially under 10) had higher survival rates")
print("4. Passengers who paid higher fares were more likely to survive")
print("5. Most passengers traveled alone (SibSp=0, Parch=0)")
print("6. Fare distribution shows many low-fare passengers and some extreme high-fare outliers")

# Additional visualizations
# Survival by age and class
plt.figure(figsize=(12, 6))
sns.violinplot(x='Pclass', y='Age', hue='Survived', data=df, split=True)
plt.title('Survival by Age and Passenger Class')
plt.show()

# Family size impact
df['FamilySize'] = df['SibSp'] + df['Parch']
plt.figure(figsize=(10, 6))
sns.barplot(x='FamilySize', y='Survived', data=df)
plt.title('Survival Rate by Family Size')
plt.show()