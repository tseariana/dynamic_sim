max_n=
max_l=
zeros= np.zeros((max_n, max_l))
for n in np.arange(max_n):
    for l in np.arange(max_l):
        lamda = float(besseljzero(l + 0.5, n))
        zeros[n, l] = lamda


