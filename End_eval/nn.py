
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup


url = "https://raw.githubusercontent.com/PIYUSH06VERMA/DataDiction/refs/heads/main/End_eval/dataset.html"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

table = soup.find("table")


headers = [th.text.strip() for th in table.find_all("th")]


rows = []
for tr in table.find_all("tr")[1:]:
    rows.append([td.text.strip() for td in tr.find_all("td")])

df = pd.DataFrame(rows, columns=headers)

print("Data scraped successfully!")
print(df.head())


df = df.apply(pd.to_numeric)


X = df.drop("Addiction_Class", axis=1).values
y = df["Addiction_Class"].values.reshape(-1, 1)


X = (X - X.mean(axis=0)) / X.std(axis=0)


np.random.seed(42)

input_size = X.shape[1]
hidden_size = 16
output_size = 1

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))


def relu(z):
    return np.maximum(0, z)

def relu_derivative(z):
    return z > 0

def forward(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    return z1, a1, z2



def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


learning_rate = 0.01
epochs = 1000

for epoch in range(epochs):
    z1, a1, y_pred = forward(X)
    loss = mse(y, y_pred)

    # Backpropagation
    dL_dy = 2 * (y_pred - y) / len(y)

    dW2 = a1.T @ dL_dy
    db2 = np.sum(dL_dy, axis=0, keepdims=True)

    da1 = dL_dy @ W2.T
    dz1 = da1 * relu_derivative(z1)

    dW1 = X.T @ dz1
    db1 = np.sum(dz1, axis=0, keepdims=True)

    # Update weights
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

print("Loss after every 100")

_, _, predictions = forward(X)

print("\nThe 5 Predictions using initial dataset:")
print(predictions[:5])
