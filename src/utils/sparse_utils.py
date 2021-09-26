# this is not used in a new versions of a model
# was used when a grah was compressed into a matrix (multi-edges were aggregated)

import numpy as np
import scipy.sparse as sparse


def multiply_row(A, row_idx, alpha, trunc=False):
    '''
    multiply values in row_idx in place
    '''

    idx_start_row = A.indptr[row_idx]
    idx_end_row = A.indptr[row_idx + 1]

    A.data[idx_start_row:idx_end_row] = (alpha *
                                         A.data[idx_start_row:idx_end_row]
                                         )
    if trunc:
        A.data[idx_start_row:idx_end_row] = np.clip(
            A.data[idx_start_row:idx_end_row], 0.0, 1.0)


def multiply_col(A, col_idx, alpha, trunc=False):
    '''
    multiply values in col_idx in place
    '''
    col_indices = A.indices == col_idx
    A.data[col_indices] = (alpha * A.data[col_indices])
    if trunc:
        A.data[col_indices] = np.clip(A.data[col_indices], 0.0, 1.0)


def prop_of_row(A):

    result = np.ones(A.shape[0])

    i = 0
    n = len(A.indptr)
    while i < n-1:
        s, e = A.indptr[i], A.indptr[i+1]
        result[i] = np.prod(A.data[s:e])
        i += 1
    if A.indptr[i] < len(A.data):
        s = A.indptr[i]
        result[i] = np.prod(A.data[s:])
    return result


def prop_of_column(A):

    result = np.ones(A.shape[1])
    col_indices = A.indices
    #    print("columns", np.unique(col_indices))

    # print(".... prop_of_column fce ", A.indices, np.unique(col_indices), A.data)

    for col_idx in np.unique(col_indices):
        current_indices = A.indices == col_idx
        #        print(" .... ", A.data[current_indices], np.prod(1 - A.data[current_indices]))
        result[col_idx] = np.prod(A.data[current_indices])
    return result


def multiply_zeros_as_ones(a, b):
    c = a.minimum(b)
    r, c = c.nonzero()

    data = np.ones(len(r))
    ones = sparse.csr_matrix((data, (r, c)), shape=a.shape)

    # get common elements
    ones_a = ones.multiply(a)
    ones_b = ones.multiply(b)

    a_dif = a - ones_a
    b_dif = b - ones_b

    result = ones_a.multiply(ones_b)
    return result + a_dif + b_dif


if __name__ == "__main__":

    a = sparse.csr_matrix((5, 5))
    b = sparse.csr_matrix((5, 5))
    c = sparse.csr_matrix((5, 5))

    a[2, 1] = 0.2
    a[1, 2] = 0.2

    b[3, 4] = 0.3
    b[4, 3] = 0.3

    c[2, 1] = 0.2
    c[1, 2] = 0.2

    N = 5
    prob_no_contact = sparse.csr_matrix((N, N))  # empty values = 1.0

    for prob in [a, b, c]:
        A = prob
        if len(A.data) == 0:
            continue
        not_a = A  # empty values = 1.0
        not_a.data = 1.0 - not_a.data
        # print(prob_no_contact.todense())
        # print(not_a.todense())
        prob_no_contact = multiply_zeros_as_ones(prob_no_contact, not_a)
        # print(prob_no_contact.todense())
        # print()

    # probability of contact (whatever layer)
    prob_of_contact = prob_no_contact
    prob_of_contact.data = 1.0 - prob_no_contact.data
#    print(prob_of_contact.todense())

    a = np.zeros((5, 5))
    b = np.zeros((5, 5))
    c = np.zeros((5, 5))

    a[2, 1] = 0.2
    a[1, 2] = 0.2

    b[3, 4] = 0.3
    b[4, 3] = 0.3

    c[2, 1] = 0.2
    c[1, 2] = 0.2

#    print()
#    print()

    prob_no_contact = np.ones((N, N))

    for prob in [a, b, c]:
        A = prob
        not_a = 1.0 - A
        # print(prob_no_contact)
        # print(not_a)
        prob_no_contact = prob_no_contact * not_a
        # print(prob_no_contact)
        # print()

    # probability of contact (whatever layer)
    prob_of_contact = 1.0 - prob_no_contact
#    print(prob_of_contact)
