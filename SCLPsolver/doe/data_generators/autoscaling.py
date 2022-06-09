import numpy as np

def generate_autoscaling_data(a, b, c, gamma, mu, alpha, q):
    """
    Generate workload data, new format.
    This function currently works for a single server with two queues for servicing two job classes.

    Parameters
    ----------
    a : list of float
        request arrival rates by service buffer (vector of size K)
    b : list of float
        cpu limits per server (vector of size I)
    c : list of float
        holding cost per unit time per buffer (vector of size K)
    gamma: list of float
        replica/cpu cost per service (vector of size J)
    mu : list of float
        mean request processing rate by service (vector of size J)
    alpha : list of float
        initial quantity of buffers (vector of size K)
    q : list of float
        maximum time in queue (vector of size J)
    Returns
    -------
    A tuple with the following values:
        G
        H
        F
        gamma
        c
        d
        alpha
        a
        b
        T
        total_buffer_cost
        cost
    """
    I = len(b)
    K = len(mu)
    J = len(gamma)

    G1 = np.diag(mu)
    G2 = np.diag(-mu)
    G = np.concatenate((G1, G2))
    H = np.ones((1,J))
    d = np.empty(0)
    T = None
    F = np.empty((2*K, 0))
    cost = np.concatenate((c, np.zeros(J)))
    alpha = np.concatenate((alpha, -alpha + a * q))
    a = np.concatenate((a, -a))
    total_buffer_cost = (np.inner(cost, alpha), np.inner(cost, a))
    c = np.matmul(cost, G)

    return G, H, F, gamma, c, d, alpha, a, b, T, total_buffer_cost, cost