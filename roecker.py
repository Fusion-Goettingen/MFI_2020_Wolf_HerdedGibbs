import numpy as np
from scipy.optimize import linear_sum_assignment


def roecker_wrapper(likelihood, num_remove=2, num_stop=10, best=False):
    """
    approximates LMB weights by computing assignments using Roecker's method.

    Parameters
    ----------
    likelihood:     LMB Likelihood
    num_remove:     max. number of single associations to remove at once
    num_stop:       break loop after num_stop iterations
    best:           compute best association (instead of greedy)

    Returns approximated LMB weights
    -------

    """
    r = likelihood.shape[0]
    c = likelihood.shape[1]
    sum_cost = - np.log(likelihood + 1e-16)

    # number of at most generated samples
    n_samples = num_stop ** (num_remove + 1)

    # allocate space
    weights = np.zeros((r, c), dtype=np.float64)
    roecker_cost = np.zeros(n_samples, dtype=np.float64)
    n_assigned = 0

    # sort single associations by cost
    idx_list_sorted = np.dstack(np.unravel_index(np.argsort(sum_cost, axis=None), sum_cost.shape))[0]

    # generate first solution
    if best:
        assignment, cost = lin_sum(sum_cost, likelihood)
    else:
        assignment, cost = greedy(idx_list_sorted, likelihood)
    print(assignment - 1)
    roecker_cost[n_assigned] += cost

    for i in range(r):
        weights[i, assignment[i]] += cost
    n_assigned += 1

    # compute associations recursively
    for i in range(1, num_remove + 1):
        n_assigned, weights = remove_assignments(-1, i, num_stop, likelihood, sum_cost, roecker_cost, idx_list_sorted,
                                                 n_assigned,
                                                 weights, best)

    # normalize weight matrix
    norm = roecker_cost[:n_assigned].sum()
    weights /= norm
    return weights


def remove_assignments(remove_after, num_remove, num_stop, likelihood, sum_cost, roecker_cost, idx_list, n_assigned,
                       weights, best):
    """
    recursively calls remove_one

    Parameters
    ----------
    remove_after:   only discard single associations in list after this index
    num_remove:     max. length of tuple to be removed
    num_stop:       break loop after num_stop iterations
    likelihood:     likelihood used to compute weights
    sum_cost:       cost matrix
    roecker_cost:   costs of samples assignments
    idx_list:       sorted list of single associations
    n_assigned:     number of assignments sampled so far
    weights:        LMB weights to be approximated
    best:           compute best association (instead of greedy)

    Returns         updated number of assignments and LMB weights
    -------

    """
    if num_remove == 1:
        n_assigned, weights = remove_one(remove_after, num_stop, likelihood, sum_cost, roecker_cost, idx_list,
                                         n_assigned, weights, best)
    else:
        for j in range(remove_after + 1, np.minimum(remove_after + 1 + num_stop, idx_list.shape[0])):
            if best:
                sum_cost_tmp = sum_cost.copy()
                sum_cost_tmp[tuple(idx_list[j])] = 1e6
            else:
                sum_cost_tmp = sum_cost

            tmp_list = idx_list[np.arange(idx_list.shape[0]) != j]

            if num_remove > 2:
                # remove after = j -1 instead of j (as in paper) because element was deleted
                n_assigned, weights = remove_assignments(j - 1, num_remove - 1, num_stop, likelihood, sum_cost_tmp,
                                                         roecker_cost, tmp_list, n_assigned, weights, best)
            else:
                n_assigned, weights = remove_one(j - 1, num_stop, likelihood, sum_cost_tmp, roecker_cost, tmp_list,
                                                 n_assigned, weights, best)
    return n_assigned, weights


def remove_one(remove_after, num_stop, likelihood, sum_cost, roecker_cost, idx_list, n_assigned, weights, best):
    """
    removes one single association and generates an assingment. Repeats until num_stop assignments are made.

    Parameters
    ----------
    remove_after:   only discard single associations in list after this index
    num_stop:       break loop after num_stop iterations
    likelihood:     likelihood used to compute weights
    sum_cost:       cost matrix
    roecker_cost:   costs of samples assignments
    idx_list:       sorted list of single associations
    n_assigned:     number of assignments sampled so far
    weights:        LMB weights to be approximated
    best:           compute best association (instead of greedy)

    Returns         updated number of assignments and LMB weights
    -------

    """
    r = weights.shape[0]
    tt_range = np.arange(r)  # indices of targets
    for i in range(remove_after + 1, np.minimum(remove_after + 1 + num_stop, idx_list.shape[0])):
        if best:
            sum_cost_tmp = sum_cost.copy()
            # remove current single association
            sum_cost_tmp[tuple(idx_list[i])] = 1e6
            # generate assignment
            assignment, cost = lin_sum(sum_cost_tmp, likelihood)
        else:
            # remove current single association
            tmp_list = idx_list[np.arange(idx_list.shape[0]) != i]
            # generate assignment
            assignment, cost = greedy(tmp_list, likelihood)

        roecker_cost[n_assigned] += cost

        # compare to existing solutions
        duplicate_found = False
        for s in range(n_assigned):
            if roecker_cost[n_assigned] == roecker_cost[s]:
                duplicate_found = True
                break
        if not duplicate_found:
            weights[tt_range, assignment] += roecker_cost[n_assigned]

            # go to next cycle
            n_assigned += 1
        print(assignment - 1)
    return n_assigned, weights


def greedy(tmp_list, likelihood):
    """
    Computes a greedy assignment.

    Parameters
    ----------
    tmp_list:   sorted list of associations
    likelihood: likelihood used to compute weights

    Returns     greedy assignment with corresponding cost
    -------

    """
    r = likelihood.shape[0]
    assignment = np.zeros(r, dtype=np.int64)  # assignment to be generated
    while tmp_list.shape[0] > 0:  # while there are single associations left
        # get best single association
        single_assoc = tmp_list[0]
        assignment[single_assoc[0]] = single_assoc[1]

        # remove all associations with same target
        tmp_list = tmp_list[tmp_list[:, 0] != single_assoc[0]]
        # remove all associations with same measurement (except missdetected / died)
        if single_assoc[1] > 1:  # exclude 0 and 1
            tmp_list = tmp_list[tmp_list[:, 1] != single_assoc[1]]

    # compute weight of assignment
    cost = 1.0
    for j in range(r):
        cost *= likelihood[j, assignment[j]]
    return assignment, cost


def lin_sum(sum_cost, likelihood):
    """
    Computes the best assignment.

    Parameters
    ----------
    sum_cost:   cost matrix
    likelihood: likelihood used to compute weights

    Returns     Best joint assignment
    -------

    """
    r, c = likelihood.shape
    # compute extended cost matrix
    tt_range = np.arange(r)
    cost_ext = np.zeros((r, r + r + c - 2), np.float64)
    cost_ext[:, :2 * r] = 1e6
    cost_ext[tt_range, tt_range] = sum_cost[:, 0]  # death
    cost_ext[tt_range, tt_range + r] = sum_cost[:, 1]  # missed detection
    cost_ext[tt_range, 2 * r:] = sum_cost[:, 2:]  # measurements

    # compute best assignment
    r_idx, assignment = linear_sum_assignment(cost_ext)

    # calculate assignment from extended assignment
    assignment[assignment < r] = 0  # deaths
    assignment[np.logical_and(assignment >= r, assignment < 2 * r)] = 1  # missed detection
    assignment[assignment >= 2 * r] += 2 - 2 * r  # measurements

    cost = 1.0
    for j in range(r):
        cost *= likelihood[j, assignment[j]]

    return assignment, cost
