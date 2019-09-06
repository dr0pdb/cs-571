import time


def profile(epsilon1, epsilon2, epsilon3, x):
    y = -1
    epsilon4 = epsilon3 + (epsilon2 - epsilon1)
    if epsilon1 <= x < epsilon2:
        y = (x - epsilon1) / (epsilon2 - epsilon1)
    elif epsilon2 <= x <= epsilon3:
        y = 1
    elif epsilon3 < x <= epsilon4:
        y = (epsilon4 - x) / (epsilon4 - epsilon3)
    return y


def fuzzication(epsilon: 'array', x):
    epsilon1 = -1 * epsilon[0]
    epsilon2 = 0
    epsilon3 = 0
    y_zero = profile(epsilon1, epsilon2, epsilon3, x)

    epsilon1 = 0
    epsilon2 = epsilon[1]
    epsilon3 = epsilon[2]
    y_small_pos = profile(epsilon1, epsilon2, epsilon3, x)

    epsilon1 = -1 * (epsilon[1] + epsilon[2])
    epsilon2 = -1 * epsilon[2]
    epsilon3 = -1 * (epsilon[1])
    y_small_neg = profile(epsilon1, epsilon2, epsilon3, x)

    return y_small_neg, y_zero, y_small_pos


def rules(theta, omega, epsilon_theta: 'array', epsilon_omega: 'array'):
    y_theta = fuzzication(epsilon_theta, theta)
    y_omega = fuzzication(epsilon_omega, omega)

    dictionary = {'00': 2, '01': 1, '02': 0, '10': 1,
                  '11': 0, '12': -1, '20': 0, '21': -1, '22': -2}
    y_curr = []
    for id1, val1 in enumerate(y_theta):
        for id2, val2 in enumerate(y_omega):
            if val1 == -1 or val2 == -1:
                continue
            else:
                curr_belongingness = min(val1, val2)
                curr_id = dictionary[str(id1) + str(id2)]
                y_curr.append([curr_belongingness, curr_id])
    return y_curr


def defuzzify(epsilon: 'array of epsilon for curr', y):
    epsilon1 = epsilon[0]
    epsilon2 = epsilon[1]
    epsilon3 = epsilon[2]
    epsilon4 = epsilon3 + (epsilon2 - epsilon1)
    x_centroid = (epsilon1 + epsilon4) / 2
    base1 = epsilon4 - epsilon1
    base2 = base1 - 2*(epsilon2 - epsilon1)
    area = 0.5 * (base1 + base2) * y
    return x_centroid, area


def compute_current(theta, omega, epsilon_theta, epsilon_omega, epsilon_curr):
    return (theta+omega)/(-5)


def main():
    epsilon_theta = [3, 2, 5]
    epsilon_omega = [2, 2, 4]
    epsilon_curr = [2, 4, 8, 6, 10, 12]
    theta = 1
    omega = 1

    current = compute_current(
        theta, omega, epsilon_theta, epsilon_omega, epsilon_curr)
    print(current)
    while True:
        time.sleep(1)
        theta_new = theta + omega / 10 + current / 200
        omega_new = omega + current / 10
        theta, omega = theta_new, omega_new
        print(theta_new, omega_new)
        current = compute_current(
            theta_new, omega_new, epsilon_theta, epsilon_omega, epsilon_curr)
        print(current)


if __name__ == "__main__":
    main()