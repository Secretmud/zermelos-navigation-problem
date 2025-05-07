import matplotlib.pyplot as plt
import numpy as np

v = 1


def S(x, D=20.0, A=0.9, B=0.15):
    current = (A - B) * np.sin(np.pi * x / D) + B
    return current


def should_accept(new_cost, current_cost, T):
    print(f"{new_cost} - {current_cost} = {new_cost - current_cost}")
    if new_cost - current_cost < 0:
        return True
    if new_cost - current_cost >= 0:
        prob = np.exp((current_cost - new_cost) / T)
        return np.random.rand() < prob


def flip_two(x):
    x = np.array(x, dtype=int)
    N = len(x)

    i1 = np.random.randint(1, N)
    i2 = np.random.randint(1, N)

    while i1 == i2:
        i2 = np.random.randint(1, N)

    def random_flip(val):
        choices = [-1, 0, 1]
        choices.remove(val)  # Exclude current value
        return np.random.choice(choices)

    x[i1] = random_flip(x[i1])
    x[i2] = random_flip(x[i2])

    return x


def time_cost(x, D=20.0):
    N = len(x)
    if np.sum(x) != 0:
        return 1e12
    dx = D / N
    total_time = 0
    for k in range(N):
        total_time += dx*(S(k*dx+dx/2)/(v**2-S(k+dx/2)))**2

    if x[0] != x[-1]:
        total_time += 1e12
    return total_time


N = 50
Sk = [0]*N
T0 = 1
gamma = 10e-3
E = time_cost(Sk)

print("Initial cost:", E)

costs = []
time = []
plt.figure(figsize=(8, 6))
plt.title('Cost Function (Time to Cross) for Random Paths')
plt.xlabel('Path Index')
plt.ylabel('Cost (Time)')
plt.ylim(top=5, bottom=5)
plt.grid(True)
x = np.linspace(0, 20, N)
T = T0*np.exp(-gamma*0)
k = 0
while T > 10e-4:
    plt.plot(x, Sk, marker='o', linestyle='-', color='b')
    Sk_new = Sk.copy()  # Generate a new solution
    Sk_new = flip_two(Sk_new)  # Apply the flip operation
    if np.sum(Sk_new) != 0:
        continue
    # Calculate the cost of the new solution
    E_new = time_cost(Sk_new)
    print(f"{k} - New cost: {E_new}, Current cost: {E}, Temperature: {T}")
    # Acceptance probability
    if should_accept(E_new, E, T):
        Sk = Sk_new  # Accept the new solution
        E = E_new  # Update the cost
        costs.append(E)

    k += 1
    T = T0*np.exp(-gamma*k)  # Decrease temperature
    # Decrease temperature
    plt.pause(0.01)  # Pause to update the plo
    plt.clf()

# Plotting the cost function
plt.show()
