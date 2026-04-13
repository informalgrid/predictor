import numpy as np
from sklearn.linear_model import LinearRegression
import pickle

# Sample data (study hours vs marks)
X = np.array([[1],[2],[3],[4],[5]])
y = np.array([20, 40, 50, 70, 90])

model = LinearRegression()
model.fit(X, y)

# Save model
pickle.dump(model, open('model.pkl', 'wb'))




