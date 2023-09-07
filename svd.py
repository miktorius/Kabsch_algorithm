import numpy as np


def golub_reinsch_svd(a):

    max_iter_test_f_splitting = 64
    eps = 2.220446049250313e-15  # can't be smaller than the machine epsilon
    # tol = machine zero / machine epsilon
    tol = 2.2250738585072014e-307 / 2.220446049250313e-16

    m, n = np.shape(a)
    e = np.zeros(n)
    q = np.zeros(n)
    v = np.zeros((n, n))
    u = np.zeros((m, n))

    for i in range(m):
        for j in range(n):
            u[i][j] = a[i][j]

    # Householder's reduction to bidiagonal form
    g = 0
    x = 0
    for i in range(n):
        e[i] = g
        s = 0
        l = i + 1
        for j in range(i, m):
            s = s + u[j][i] * u[j][i]
        if s < tol:
            g = 0
        else:
            f = u[i][i]
            g = np.sqrt(s) if f < 0 else -np.sqrt(s)
            h = f * g - s
            u[i][i] = f - g
            for j in range(l, n):
                s = 0
                for k in range(i, m):
                    s = s + u[k][i] * u[k][j]
                f = s / h
                for k in range(i, m):
                    u[k][j] = u[k][j] + f * u[k][i]
        q[i] = g
        s = 0
        for j in range(l, n):
            s = s + u[i][j] * u[i][j]
        if s < tol:
            g = 0
        else:
            f = u[i][i+1]
            g = np.sqrt(s) if f < 0 else -np.sqrt(s)
            h = f * g - s
            u[i][i+1] = f - g
            for j in range(l, n):
                e[j] = u[i][j] / h
            for j in range(l, m):
                s = 0
                for k in range(l, n):
                    s = s + u[j][k] * u[i][k]
                for k in range(l, n):
                    u[j][k] = u[j][k] + s * e[k]
        y = abs(q[i]) + abs(e[i])
        if y > x:
            x = y

    # Accumulation of right-hand transformations
    for i in range(n - 1, -1, -1):
        if g != 0:
            h = u[i][i+1] * g
            for j in range(l, n):
                v[j][i] = u[i][j] / h
            for j in range(l, n):
                s = 0
                for k in range(l, n):
                    s = s + u[i][k] * v[k][j]
                for k in range(l, n):
                    v[k][j] = v[k][j] + s * v[k][i]
        for j in range(l, n):
            v[i][j] = 0
            v[j][i] = 0
        v[i][i] = 1
        g = e[i]
        l = i

    # Accumulation of left-hand transformations
    for i in range(n - 1, -1, -1):
        l = i + 1
        g = q[i]
        for j in range(l, n):
            u[i][j] = 0
        if g != 0:
            h = u[i][i] * g
            for j in range(l, n):
                s = 0
                for k in range(l, m):
                    s = s + u[k][i] * u[k][j]
                f = s / h
                for k in range(i, m):
                    u[k][j] = u[k][j] + f * u[k][i]
            for j in range(i, m):
                u[j][i] = u[j][i] / g
        else:
            for j in range(i, m):
                u[j][i] = 0
        u[i][i] = u[i][i] + 1

    # Diagonalization of the bidiagonal form
    eps = eps * x
    for k in range(n - 1, -1, -1):
        for iter in range(max_iter_test_f_splitting):

            # Test f splitting
            test_f_convergence = False
            for l in range(k, -1, -1):
                if abs(e[l]) <= eps:
                    test_f_convergence = True
                    break
                if abs(q[l-1]) <= eps:
                    break

            # Cancellation of e[l] if l > 1
            if test_f_convergence == False:
                c = 0
                s = 1
                l1 = l-1
                for i in range(l, k+1):
                    f = s * e[i]
                    e[i] = c * e[i]
                    if abs(f) <= eps:
                        break
                    g = q[i]
                    h = np.sqrt(f * f + g * g)
                    q[i] = h
                    c = g / h
                    s = -f / h
                    for j in range(m):
                        y = u[j][l1]
                        z = u[j][i]
                        u[j][l1] = y * c + z * s
                        u[j][i] = -y * s + z * c

            # Test f convergence
            z = q[k]
            convergence = False
            if l == k:
                convergence = True

            if convergence == False:
                # Shift from bottom 2 x 2 minor
                x = q[l]
                y = q[k-1]
                g = e[k-1]
                h = e[k]
                f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y)
                g = np.sqrt(f * f + 1)
                f = ((x - z) * (x + z) + h *
                     (y / (f-g if f < 0 else f+g) - h)) / x

                # Next QR transformation
                c = 1
                s = 1
                for i in range(l+1, k+1):
                    g = e[i]
                    y = q[i]
                    h = s * g
                    g = c * g
                    e[i-1] = np.sqrt(f * f + h * h)
                    z = e[i-1]
                    c = f / z
                    s = h / z
                    f = x * c + g * s
                    g = -x * s + g * c
                    h = y * s
                    y = y * c
                    for j in range(n):
                        x = v[j][i-1]
                        z = v[j][i]
                        v[j][i-1] = x * c + z * s
                        v[j][i] = -x * s + z * c
                    q[i-1] = np.sqrt(f * f + h * h)
                    z = q[i-1]
                    c = f / z
                    s = h / z
                    f = c * g + s * y
                    x = -s * g + c * y
                    for j in range(m):
                        y = u[j][i-1]
                        z = u[j][i]
                        u[j][i-1] = y * c + z * s
                        u[j][i] = -y * s + z * c
                e[l] = 0
                e[k] = f
                q[k] = x

            if convergence == True:
                if z < 0:
                    # q[k] is made non-negative
                    q[k] = -z
                    for j in range(n):
                        v[j][k] = -v[j][k]
                # Breaking current k cycle
                break
            elif iter == max_iter_test_f_splitting-1:
                print("No convergence")
                exit()

    U = u.T
    S = np.diag(q)
    V = v.T

    # Sorting singular values of SIGMA and placing singular vectors if U and V in corresponding order using bubble sort
    for i in range(len(S)):
        for j in range(len(S) - 1):
            if S[j][j] < S[j + 1][j + 1]:
                S[j][j], S[j+1][j+1] = S[j+1][j+1], S[j][j]

                temp = U[j].copy()
                U[j] = U[j+1].copy()
                U[j+1] = temp

                temp = V[j].copy()
                V[j] = V[j+1].copy()
                V[j+1] = temp

    return U.T, S, V.T
