import numpy as np
from scipy.stats import multivariate_normal
from herded_gibbs import herded_gibbs_assign
from murty import murty_wrapper
from roecker import roecker_wrapper

Z_k = np.array([
    [1.25, 2.4],
    [1.65, 1.3],
    [3.2, 3.4],
    [5.4, 2.5]
])
dt_tt = np.dtype([('l', 'i8'), ('r', 'f8'), ('GM', 'O')])
dt_GM = np.dtype([('m', 'f8', 2), ('P', 'f8', (2, 2)), ('w', 'f8')])
tt_lmb_predicted = np.zeros(4, dtype=dt_tt)
for t in tt_lmb_predicted:
    t['GM'] = np.zeros(1, dtype=dt_GM)
tt_lmb_predicted['l'] = np.arange(1, 5)
tt_lmb_predicted['r'] = np.array([1.0, 0.6, 1.0, 0.9])
P = np.diag([1, 1])
tt_lmb_predicted[0]['GM'] = np.array([
    ([1.5, 2.], P, 1.)
], dtype=dt_GM)
tt_lmb_predicted[1]['GM'] = np.array([
    ([3.5, 1.], P, 1.)
], dtype=dt_GM)
tt_lmb_predicted[2]['GM'] = np.array([
    ([5.5, 3.], P, 1.)
], dtype=dt_GM)
tt_lmb_predicted[3]['GM'] = np.array([
    ([6., 2.], P, 1.)
], dtype=dt_GM)

H = np.array([[1, 0], [0, 1]])
D = np.diag([0.5, 0.5])
R = D @ D.T
P_D = 0.95
Q_D = 1 - P_D
pdf_c = 4 / (7 * 4)

num_Z = Z_k.shape[0]
num_T = tt_lmb_predicted.shape[0]

tt_single_update = np.zeros((num_T, num_Z + 1), dtype=dt_tt)

eta_Z = np.zeros((num_T, num_Z + 1))

# compute single likelihood and updated components for all measurements
for z in range(num_Z):

    for (i, track) in enumerate(tt_lmb_predicted):
        cmpnt_update = np.zeros(track['GM'].shape[0], dtype=dt_GM)

        for (j, cmpnt) in enumerate(track['GM']):
            # Kalman update
            z_plus = H @ cmpnt['m']
            S = H @ cmpnt['P'] @ H.T + R
            S_inv = np.linalg.inv(S)
            K = cmpnt['P'] @ H.T @ S_inv

            # update mean and covariance

            cmpnt_update[j]['m'] = cmpnt['m'] + K @ (Z_k[z] - z_plus)
            cmpnt_update[j]['P'] = cmpnt['P'] - K @ S @ K.T
            cmpnt_update[j]['w'] = cmpnt['w'] * multivariate_normal.pdf(Z_k[z], mean=z_plus,
                                                                        cov=S) * P_D / pdf_c + 1e-24

        # likelihood as sum of GM weights
        sum_w = np.sum(cmpnt_update['w'])
        cmpnt_update['w'] /= sum_w

        eta_Z[i, z + 1] = sum_w
        tt_single_update[i, z + 1]['GM'] = cmpnt_update

# missed detection case
eta_Z[:, 0] = Q_D

likelihood = np.zeros((eta_Z.shape[0], eta_Z.shape[1] + 1))
likelihood[:, 1:] = tt_lmb_predicted['r'][:, None] * eta_Z
likelihood[:, 0] = 1 - tt_lmb_predicted['r']

likelihood = np.round(likelihood,1)
print(likelihood)

print("#########    Herded Gibbs      #########")
weights_hg = herded_gibbs_assign(likelihood, n_samples=5)
print("#########    Murty      #########")
weights_murty = murty_wrapper(likelihood, num_samples=5)
print("#########    Roecker      #########")
weights_roecker = roecker_wrapper(likelihood, num_remove=1, best=False)



aaaah = 42

