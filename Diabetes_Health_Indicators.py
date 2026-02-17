import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier  

df = pd.read_csv(r"C:\Users\jessi\Downloads\Diabetes_Health_Indicators\Diabetes_data.csv")

print("Data Info:\n")
print(df.info())


print("\nData Description:\n")
print(df.describe())

print("First 5 rows of the dataset:")
print(df.head())
df['Gender'] = df['Gender'].str.strip().str.lower().replace({
    'male': 'Male', 'malemale': 'Male', 'female': 'Female', 'femalefemale': 'Female'
})

df.drop_duplicates(inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].str.lower().str.strip().fillna(df[col].mode()[0])

for col in df.select_dtypes(include=['int64', 'float64']).columns:
    df[col] = df[col].fillna(df[col].median())

if 'Age' in df.columns:
    df = df[df['Age'] >= 0]

label_cols = ['Gender', 'Family History', 'Smoking', 'Physically Active', 
              'High Cholesterol', 'Stroke History', 'Heart Disease']
le = LabelEncoder()
for col in label_cols:
    df[col] = le.fit_transform(df[col])

df = df.dropna(subset=['Diabetes'])

plt.figure(figsize=(8, 6))
plt.hist(df['Age'], bins=20, color='skyblue', edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

Gender_count = df['Gender'].value_counts()
plt.figure(figsize=(8, 6))
plt.pie(Gender_count, labels=Gender_count.index, autopct="%0.2f%%", startangle=90, colors=['lightblue', 'pink'])
plt.title('Gender Distribution (Pie Chart)')
plt.axis('equal')
plt.show()

number_columns = df.select_dtypes(include='number').columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[number_columns].corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='Gender', hue='Diabetes')
plt.title("Diabetes by Gender")
plt.xlabel("Gender")
plt.ylabel("Count")
plt.legend(title="Diabetes")
plt.show()

X = df.drop('Diabetes', axis=1)
y = df['Diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model_rf = RandomForestClassifier(n_estimators=500, max_depth=25, random_state=42, class_weight='balanced')
model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print("Random Forest Accuracy:", round(accuracy_rf * 100, 2), "%")

model_gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=10, random_state=42)
model_gb.fit(X_train, y_train)
y_pred_gb = model_gb.predict(X_test)
accuracy_gb = accuracy_score(y_test, y_pred_gb)
print("Gradient Boosting Accuracy:", round(accuracy_gb * 100, 2), "%")

if accuracy_gb > accuracy_rf:
    best_model = model_gb
    print("Using Gradient Boosting as the best model.")
else:
    best_model = model_rf
    print("Using Random Forest as the best model.")

columns = ['Age', 'Gender', 'BMI', 'Glucose Level', 'Blood Pressure',
           'Family History', 'Smoking', 'Physically Active',
           'High Cholesterol', 'Stroke History', 'Heart Disease']
sample_input = pd.DataFrame([[35, 1, 28.5, 130, 80, 1, 0, 1, 0, 0, 1]], columns=columns)
prediction = best_model.predict(sample_input)
print("Prediction for sample input:", "Yes" if prediction[0] == 1 else "No")