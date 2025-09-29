import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import random

# -----------------------------
# Step 1: Creating Synthetic Data
# -----------------------------
def create_traffic_data(n_samples=5000):
    data = []
    for _ in range(n_samples):
        # Cars waiting in each direction
        north = random.randint(0, 50)
        south = random.randint(0, 50)
        east = random.randint(0, 50)
        west = random.randint(0, 50)

        # The "label" = which direction should get green light
        # Rule of thumb: pick the direction with the max amount of cars
        decision = np.argmax([north, south, east, west])

        data.append([north, south, east, west, decision])
    return pd.DataFrame(data, columns=["north", "south", "east", "west", "decision"])

traffic_data = create_traffic_data()

# -----------------------------
# Step 2: Train the model
# -----------------------------
X = traffic_data[["north", "south", "east", "west"]]
y = traffic_data["decision"]

model = DecisionTreeClassifier()
model.fit(X, y)

# -----------------------------
# Step 3: Test the model
# -----------------------------
test_case = np.array([[10, 45, 12, 7]])  # (north=10, south=45, east=12, west=7)
prediction = model.predict(test_case)[0]

directions = ["North", "South", "East", "West"]
print(f"Cars: {test_case.tolist()[0]}")
print(f"ML Model Suggests: Give GREEN light to {directions[prediction]}")

# -----------------------------
# Step 4: Simulation loop
# -----------------------------
def simulate_step(model, state):
    """
    state = [north, south, east, west]
    """
    action = model.predict([state])[0]
    state[action] = max(0, state[action] - random.randint(5, 15))  # cars clear
    # new cars arrive randomly
    state = [x + random.randint(0, 5) for x in state]
    return state, action

# Run simulation
state = [30, 20, 25, 15]
for step in range(5):
    state, action = simulate_step(model, state)
    print(f"Step {step+1} -> State: {state}, Green: {directions[action]}")






