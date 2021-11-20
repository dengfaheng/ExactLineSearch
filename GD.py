import time

import numpy as np

n = pow(10, 6)
m = 100


def is_pos_def(x):
    # return np.all(np.linalg.eigvals(x) > 0)
    return x > 0


def is_neg_def(x):
    # return ~np.all(np.linalg.eigvals(x) > 0)
    return x < 0


def is_zero_def(x):
    # return np.all(np.linalg.eigvals(x) == 0)
    return x == 0


def bi_section(X: np.array, y: np.array, beta: np.array, decent: np.array, upper=0.000001, tolerance=0.000000001) -> float:
    curr_lower = 0.0
    curr_upper = upper
    alpha_curr = 0.0
    while (curr_upper - curr_lower) > tolerance:
        alpha_curr = (curr_upper + curr_lower) / 2
        print("upper ", curr_upper)
        print("lower ", curr_lower)
        print("alpha ", alpha_curr)
        print("\n")
        # alpha_hessian = (X.T.dot(X).dot(beta + alpha_curr * decent) - 2 * X.T.dot(y)).dot(decent.T)
        # alpha_hessian = decent.dot((X.T.dot(X).dot(beta + alpha_curr * decent) - 2 * X.T.dot(y)))
        l2_sum = 0
        for i in range(m):
            X_i = X[[i]]
            # print(X_i)
            y_i = y[[i]]
            # print(y_i)
            # y_i = y_i[:[0]]
            l2_sum += (X_i.dot(beta - alpha_curr * decent) - y_i).dot(X_i.dot(decent))
            # print("l2_sum", l2_sum)
        l2_sum = l2_sum * (-2)

        if is_zero_def(l2_sum):
            break
        if is_pos_def(l2_sum):
            curr_upper = alpha_curr
        if is_neg_def(l2_sum):
            curr_lower = alpha_curr
    return alpha_curr


# (exact line search)
def gradient_descent(X: np.array, y: np.array, iterations=100, stop_condition=0.001):
    total_n = X.shape[1]
    beta_curr = np.zeros((total_n, 1))

    last_cost = -100.0
    for iteration in range(iterations):
        y_predicted = X.dot(beta_curr)
        # cost = sum([val ** 2 for val in (y_predicted - y)])
        cost = np.linalg.norm(y_predicted - y, ord=2) ** 2
        # stop here?
        if abs(cost - last_cost) < stop_condition:
            break
        beta_decent = 2 * X.T.dot(X).dot(beta_curr) - 2 * X.T.dot(y)
        learning_alpha = bi_section(X, y, beta_curr, beta_decent)
        # learning_alpha = 0.0001
        beta_curr = beta_curr - learning_alpha * beta_decent
        print("iteration {}, cost = {}".format(iteration, round(cost, 6)))
        last_cost = cost


# end def


the_x = np.random.rand(m, n)
the_y = np.random.rand(m, 1)

if __name__ == '__main__':
    print("hello")
    time_start = time.time()
    gradient_descent(the_x, the_y, iterations=10000)
    time_end = time.time()
    print("total time = ", time_end-time_start)