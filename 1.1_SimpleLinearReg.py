# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Get dataset
df_sal = pd.read_csv(r'F:\coding\AI\BootCamp\Salary_Data.csv')
print (df_sal.head())

# Describe data
print (df_sal.describe())

# Data distribution
plt.title('Salary Distribution Plot')
sns.histplot(df_sal['Salary'])
plt.show()

# Splitting variables
X = df_sal.iloc[:, :1]  # independent
y = df_sal.iloc[:, 1:]  # dependent
     
# Relationship between Salary and Experience
plt.figure(figsize=(8,5))  # تنظیم اندازه نمودار
plt.scatter(df_sal['YearsExperience'], df_sal['Salary'], color='lightcoral', s=80, alpha=0.7)
plt.title('Salary vs Experience', fontsize=14, fontweight='bold')
plt.xlabel('Years of Experience', fontsize=12)
plt.ylabel('Salary', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)  # اضافه کردن خطوط شبکه
plt.box(False)  # حذف کادر اطراف نمودار
plt.show()
     
# Splitting dataset into test/train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Regressor model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Prediction result
y_pred_test = regressor.predict(X_test)     # predicted value of y_test
y_pred_train = regressor.predict(X_train)   # predicted value of y_train



# Prediction on training set
plt.scatter(X_train, y_train, color = 'lightcoral')
plt.plot(X_train, y_pred_train, color = 'firebrick')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['X_train/Pred(y_train)', 'X_train/y_train'], title = 'Sal/Exp', loc='best', facecolor='white')
plt.box(False)
plt.show()
     
     

# Prediction on test set
plt.scatter(X_test, y_test, color = 'lightcoral')
plt.plot(X_test, y_pred_test, color = 'firebrick')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.legend(['X_train/Pred(y_test)', 'X_test/y_test'], title = 'Sal/Exp', loc='best', facecolor='white')
plt.box(False)
plt.show()
          
# Regressor coefficients and intercept
print(f'Coefficient: {regressor.coef_}')
print(f'Intercept: {regressor.intercept_}')
     