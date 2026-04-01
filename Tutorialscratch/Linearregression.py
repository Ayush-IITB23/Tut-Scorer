import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


df = pd.read_csv('tutorials.csv')

x = df[['Duration_Minutes', 'View_Count', 'Like_Count', 'Comment_Count','Engagement_Rate', 'Channel_Subscriber_Count']]
y = df['Actual_Quality_Score']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

coefficients = pd.DataFrame({'Feature': x.columns, 'Weight': model.coef_})
coefficients = coefficients.sort_values(by="Weight")
print("\nFeature Weights:")
print(coefficients)

df['Predicted_Quality_Score'] = model.predict(x)
df['Predicted_Quality_Score'] = df['Predicted_Quality_Score'].clip(lower=0, upper=100)

top_10_tutorials = df.sort_values(by='Predicted_Quality_Score', ascending=False).head(10)
print("\nTop 10 Best Tutorials")
print(top_10_tutorials[['Video_Title','View_Count','Predicted_Quality_Score', 'Actual_Quality_Score']])

print("Intercept:", model.intercept_)
for feature, weight in zip(x.columns, model.coef_):
    print(f"{feature}: {weight}")