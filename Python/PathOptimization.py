import json
import numpy as np
import matplotlib.pyplot as plt

# ------------------------ Load data ------------------------------------------
def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    x = np.array([p["x"] for p in data])
    y = np.array([p["y"] for p in data])
    return np.column_stack((x, y))

# ---------------- Polyline gradient computation ------------------------------
def polyline_gradient(S, A, B, t):
    # S: starting point
    # A,B: segments of the rout (A-B)
    # t: normalized segment position (0,1 -> A,B)
    
    n = A.shape[0] #Length of data points
    P = A + (B - A) * t[:, np.newaxis] #Compute the internal point for each segment based on the parameter t
    dL_dt = np.zeros(n) # Initialize the derivative

    # Iterate through each segment i
    for i in range(n):
        Pi = P[i]
        dPi = B[i] - A[i] #Segment vector

        # First or previous segment
        if i == 0:
            prev = S
        else:
            prev = P[i - 1]
       
        dL_dt_prev = np.dot((Pi - prev) / np.linalg.norm(Pi - prev), dPi)

        # Next or last segment
        if i < n - 1:
            next_ = P[i + 1]
            dL_dt_next = np.dot((Pi - next_) / np.linalg.norm(Pi - next_), dPi)
        else:
            dL_dt_next = 0
        # Compute the derivative

        dL_dt[i] = dL_dt_prev + dL_dt_next

    return dL_dt

# ------------------------ Main -----------------------------------------------
nodes = load_data('nodes.json') #load nodes data for each segment
SG = load_data('from_to.json') # load start-goal data

plt.figure(1)
plt.clf()
plt.scatter(SG[:, 0], SG[:, 1], c='r', s=100, marker='X', linewidths=2)
for i in range(0, len(nodes), 2):
    plt.plot(nodes[i:i+2, 0], nodes[i:i+2, 1], '-*', color='k')
plt.ylim(0, 20)

A_nodes = np.vstack([nodes[::2], SG[1]])
B_nodes = np.vstack([nodes[1::2], SG[1]])
n_nodes = A_nodes.shape[0]

P = np.zeros((n_nodes, 2))
t = np.ones(n_nodes)*0.5 #Initialization of nodes parameters

# Gradient descent setup
G = 0.02 #Correction gain
tol = 1e-3 # Stop tolerance
dL = 100 #Initialization to a large value
L_ans = 100 #Initialization to a large value
iteration = 0 #To count the number of iterations

while dL > tol:
    dL_dt = polyline_gradient(SG[0], A_nodes, B_nodes, t)
    t -= G * dL_dt
    t = np.clip(t, 0, 1)
    P = A_nodes + (B_nodes - A_nodes) * t[:, np.newaxis]

    L = np.sum(np.linalg.norm(np.diff(np.vstack([SG[0], P]), axis=0), axis=1))
    dL = abs(L - L_ans)
    L_ans = L
    iteration += 1

    # Plotting each iteration
    plt.figure(1)
    plt.clf()
    plt.scatter(SG[:, 0], SG[:, 1], c='r', s=100, marker='X', linewidths=2)
    for i in range(0, len(nodes), 2):
        plt.plot(nodes[i:i+2, 0], nodes[i:i+2, 1], '-*', color='k')
    plt.ylim(0, 20)
    plt.plot(P[:, 0], P[:, 1], 'ro--')
    plt.title(f'Iter: {iteration}, Length: {L:.3f}')
    plt.pause(0.2)

print(f"Final length: {L_ans:.4f} after {iteration} iterations.")


# Compare the solution to the Cpp-code
# P_C_opt = load_data('C:/Users\carlo\Documents\Carlos\Career\Resume\Applications\May_2025\Orca_2025\ProgramChallenge\PolylineOptimizer\PolylineOptimizer\P.json')
# plt.plot(P_C_opt[:, 0], P_C_opt[:, 1], 'b*--')
# plt.xlim(25, 32)
# plt.ylim(0, 13)
# plt.show()

