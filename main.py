import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score


# student dataset
data = pd.read_csv('StudentsPerformance.csv')



def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)
    for i in range(n):
        x = points.iloc[i]["reading score"] #hours spent studying
        y = points.iloc[i]["math score"] #final exam score
        # Calculate gradients using the current values of m and b
        # m_gradient = -(2/n) * Σx(y - (mx + b))
        # b_gradient = -(2/n) * Σ(y - (mx + b))
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))
     
    m = m_now - (L * m_gradient) # Update m using the learning rate and gradient
    b = b_now - (L * b_gradient) # Update b using the learning rate and gradient

    return m, b # Return the updated values of m and b

m = 0
b = 0
learning_rate = 0.00001
epochs = 100#iterations


for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch {i}")
    m, b = gradient_descent(m, b, data, learning_rate)

print(f"m: {m}, b: {b}")

y_true = data["math score"]
y_pred = [m * x + b for x in data["reading score"]]

print("MSE:", mean_squared_error(y_true, y_pred))
print("R²:", r2_score(y_true, y_pred))



plt.scatter(data["reading score"], data["math score"], color='black')
plt.plot(list(range(20, 100)), [m * x + b for x in range(20, 100)], color='red')
plt.show()


