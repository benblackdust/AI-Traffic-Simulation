import traci
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Load your trained ML model (from earlier code)
# (Here we just re-train quickly for demo)
X = [[10, 20, 5, 3], [2, 5, 20, 15], [25, 10, 30, 5]]
y = [1, 2, 0]  # which direction gets green
model = DecisionTreeClassifier().fit(X, y)

# SUMO setup
sumoBinary = "sumo"  # or "sumo-gui"
sumoCmd = [sumoBinary, "-c", "memwork.sumo.cfg"]

traci.start(sumoCmd)

# Map directions to SUMO lanes
directions = ["north_in", "south_in", "east_in", "west_in"]
traffic_light_id = "TL1"  # ID from your .net.xml

for step in range(1000):  # run simulation steps
    traci.simulationStep()

    # Count cars waiting at each incoming lane
    state = []
    for lane in directions:
        state.append(traci.lane.getLastStepVehicleNumber(lane))

    # ML model decides which direction should get green
    action = model.predict([state])[0]
    green_phase = action  # map to SUMO traffic light phase

    # Apply decision: set traffic light
    traci.trafficlight.setPhase(traffic_light_id, green_phase)

    print(f"Step {step}: State={state}, Green={directions[action]}")

traci.close()