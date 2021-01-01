import numpy as np
cimport numpy as np
from numpy cimport ndarray
cimport cython

ctypedef np.float32_t DTYPE_t

ORACLE_PRECOMPUTED_TABLE = {}

@cython.boundscheck(False)
def decode(int force_gold, int sentence_len, np.ndarray[DTYPE_t, ndim=3] label_scores_chart, np.ndarray[DTYPE_t, ndim=2] type_scores_chart, int is_train, gold, label_vocab, type_vocab):
    cdef DTYPE_t NEG_INF = -np.inf

    # Label scores chart is copied so we can modify it in-place for augmentated decode
    cdef np.ndarray[DTYPE_t, ndim=3] label_scores_chart_copy = label_scores_chart.copy()
    cdef np.ndarray[DTYPE_t, ndim=2] type_scores_chart_copy = type_scores_chart.copy()

    cdef np.ndarray[DTYPE_t, ndim=3] value_one_chart = np.zeros((sentence_len+1, sentence_len+1, sentence_len+1), dtype=np.float32)
    cdef np.ndarray[DTYPE_t, ndim=3] value_muti_chart = np.zeros((sentence_len+1, sentence_len+1, sentence_len+1), dtype=np.float32)

    cdef np.ndarray[int, ndim=3] split_idx_chart = np.zeros((sentence_len+1, sentence_len+1, sentence_len+1), dtype=np.int32)
    cdef np.ndarray[int, ndim=3] best_label_chart = np.zeros((sentence_len+1, sentence_len+1, 2), dtype=np.int32)
    cdef np.ndarray[int, ndim=3] head_chart = np.zeros((sentence_len+1, sentence_len+1, sentence_len+1), dtype=np.int32)
    cdef np.ndarray[int, ndim=3] father_chart = np.zeros((sentence_len+1, sentence_len+1, sentence_len+1), dtype=np.int32)


    cdef int length
    cdef int left
    cdef int right

    cdef int child_l
    cdef int child_r
    cdef int child_head
    cdef int child_type
    cdef int type_id


    cdef np.ndarray[DTYPE_t, ndim=1] label_scores_for_span

    cdef int oracle_label_index
    cdef int oracle_type_index
    cdef DTYPE_t label_score_one
    cdef DTYPE_t label_score_empty
    cdef DTYPE_t dep_score
    cdef int argmax_label_index
    cdef int argmax_type_index
    cdef DTYPE_t left_score
    cdef DTYPE_t right_score
    cdef DTYPE_t type_max_score

    cdef int best_split
    cdef int split_idx # Loop variable for splitting
    cdef DTYPE_t split_val # best so far
    cdef DTYPE_t max_split_val

    cdef int label_index_iter, head, father

    if not force_gold:

        for length in range(1, sentence_len + 1):
            for left in range(0, sentence_len + 1 - length):
                right = left + length

                if is_train :
                    oracle_label_index = label_vocab.index(gold.oracle_label(left, right))

                    # augment: here we subtract 1 from the oracle label
                    label_scores_chart_copy[left, right, oracle_label_index] -= 1

                    # We do argmax ourselves to make sure it compiles to pure C
                    #no empty label
                argmax_label_index = 1
                if length == 1 or length == sentence_len:
                    argmax_label_index = 2 #sub_head label can not be leaf

                label_score_one = label_scores_chart_copy[left, right, argmax_label_index]
                for label_index_iter in range(argmax_label_index, label_scores_chart_copy.shape[2]):
                    if label_scores_chart_copy[left, right, label_index_iter] > label_score_one:
                        argmax_label_index = label_index_iter
                        label_score_one = label_scores_chart_copy[left, right, label_index_iter]
                best_label_chart[left, right, 1] = argmax_label_index

                label_score_empty = label_scores_chart_copy[left, right,0]

                if is_train:
                    # augment: here we add 1 to all label scores
                    label_score_one +=1
                    label_score_empty += 1

                if length == 1:
                    #head is right, index from 1
                    value_one_chart[left, right, right] = label_score_one
                    value_muti_chart[left, right, right] = label_score_empty
                    if value_one_chart[left, right, right] > value_muti_chart[left, right, right]:
                        value_muti_chart[left, right, right] = value_one_chart[left, right, right]
                        best_label_chart[left, right,0] = best_label_chart[left, right,1]
                    else:
                        best_label_chart[left, right,0] = 0 #empty label
                    head_chart[left, right, right] = -1

                    continue

                #head also in the empty part
                for head_l in range(left + 1, right + 1):
                    value_one_chart[left, right, head_l] = NEG_INF

                for split_idx in range(left + 1, right):
                    for head_l in range(left + 1, split_idx + 1):
                        for head_r in range(split_idx + 1, right + 1):

                            #head in the right empty part, left father is right
                            #left is one, right is multi
                            dep_score = type_scores_chart_copy[head_l, head_r]
                            if split_idx - left == 1:#leaf can be empty
                                split_val = value_muti_chart[left, split_idx, head_l] + value_muti_chart[split_idx, right, head_r] + dep_score
                            else :
                                split_val = value_one_chart[left, split_idx, head_l] + value_muti_chart[split_idx, right, head_r] + dep_score
                            if split_val > value_one_chart[left, right, head_r]:
                                value_one_chart[left, right, head_r] = split_val
                                split_idx_chart[left, right, head_r] = split_idx
                                head_chart[left, right, head_r] = head_l

                            #head in the left empty part, right father is left
                            #left is multi, right is one
                            dep_score = type_scores_chart_copy[head_r, head_l]
                            if right - split_idx == 1:#leaf can be empty
                                split_val = value_muti_chart[split_idx, right, head_r] + value_muti_chart[left, split_idx, head_l] + dep_score
                            else:
                                split_val = value_one_chart[split_idx, right, head_r] + value_muti_chart[left, split_idx, head_l] + dep_score
                            if split_val > value_one_chart[left, right, head_l]:
                                value_one_chart[left, right, head_l] = split_val
                                split_idx_chart[left, right, head_l] = split_idx
                                head_chart[left, right, head_l] = head_r

                for head_l in range(left + 1, right + 1):
                    if label_score_one > label_score_empty:
                        value_muti_chart[left, right, head_l] = value_one_chart[left, right, head_l] + label_score_one
                    else :
                        value_muti_chart[left, right, head_l] = value_one_chart[left, right, head_l] + label_score_empty
                    value_one_chart[left, right, head_l] = value_one_chart[left, right, head_l] + label_score_one

                if label_score_one < label_score_empty:
                    best_label_chart[left, right, 0] = 0
                else:
                    best_label_chart[left, right,0] = best_label_chart[left, right,1]
                    #add mergein

    # Now we need to recover the tree by traversing the chart starting at the
    # root. This iterative implementation is faster than any of my attempts to
    # use helper functions and recursion

    # All fully binarized trees have the same number of nodes
    cdef int num_tree_nodes = 2 * sentence_len - 1
    cdef np.ndarray[int, ndim=1] included_i = np.empty(num_tree_nodes, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] included_j = np.empty(num_tree_nodes, dtype=np.int32)

    cdef np.ndarray[int, ndim=1] included_label = np.empty(num_tree_nodes, dtype=np.int32)

    cdef np.ndarray[int, ndim=1] included_type = np.empty(sentence_len, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] included_father = np.zeros(sentence_len, dtype=np.int32)# 0 is root

    cdef int idx = 0
    cdef int stack_idx = 1
    # technically, the maximum stack depth is smaller than this
    cdef np.ndarray[int, ndim=1] stack_i = np.empty(num_tree_nodes + 5, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] stack_j = np.empty(num_tree_nodes + 5, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] stack_head = np.empty(num_tree_nodes + 5, dtype=np.int32)

    cdef np.ndarray[int, ndim=1] stack_type = np.empty(num_tree_nodes + 5, dtype=np.int32)

    cdef int i, j, k, root_head, nodetype, sub_head
    if not force_gold:
        max_split_val = NEG_INF
        for idxx in range(sentence_len):
            split_val = value_one_chart[0, sentence_len, idxx + 1] + type_scores_chart[idxx + 1, 0]
            if split_val > max_split_val:
                max_split_val = split_val
                root_head = idxx + 1
    else:
        root_head = gold.oracle_head(0, sentence_len)
    stack_i[1] = 0
    stack_j[1] = sentence_len
    stack_head[1] = root_head
    stack_type[1] = 1

    while stack_idx > 0:

        i = stack_i[stack_idx]
        j = stack_j[stack_idx]
        head = stack_head[stack_idx]
        nodetype = stack_type[stack_idx]
        stack_idx -= 1

        included_i[idx] = i
        included_j[idx] = j
        if force_gold:
            included_label[idx] = label_vocab.index(gold.oracle_label(i,j))
        else :
            if i + 1 == j:
                nodetype = 0
            included_label[idx] = best_label_chart[i, j, nodetype]

        idx += 1
        if i + 1 < j:

            if force_gold:
                oracle_splits = gold.oracle_splits(i, j)
                if head > min(oracle_splits): #head index from 1
                        #h in most right, so most left is noempty
                    k = min(oracle_splits)
                    sub_head = gold.oracle_head(i, k)
                    included_father[sub_head - 1] = head
                else:
                    k = max(oracle_splits)
                    sub_head = gold.oracle_head(k, j)
                    included_father[sub_head - 1] = head
            else:
                k = split_idx_chart[i, j, head]
                sub_head = head_chart[i,j, head]
                included_father[sub_head - 1] = head

            stack_idx += 1
            stack_i[stack_idx] = k
            stack_j[stack_idx] = j
            if head > k:
                stack_head[stack_idx] = head
                stack_type[stack_idx] = 0
            else :
                stack_head[stack_idx] = sub_head
                stack_type[stack_idx] = 1
            stack_idx += 1
            stack_i[stack_idx] = i
            stack_j[stack_idx] = k
            if head > k:
                stack_head[stack_idx] = sub_head
                stack_type[stack_idx] = 1
            else :
                stack_head[stack_idx] = head
                stack_type[stack_idx] = 0

    cdef DTYPE_t running_total = 0.0
    for idx in range(num_tree_nodes):
        running_total += label_scores_chart[included_i[idx], included_j[idx], included_label[idx]]

    for idx in range(sentence_len):
        #root_head father is 0
        if force_gold:
            argmax_type_index = type_vocab.index(gold.oracle_type(idx, idx + 1))
        else :
            argmax_type_index = 0
        #root_head father is 0
        running_total += type_scores_chart[idx + 1, included_father[idx]]
        included_type[idx] = argmax_type_index

    cdef DTYPE_t score = value_one_chart[0, sentence_len, root_head] + type_scores_chart[root_head, 0]
    if force_gold:
        score = running_total
    cdef DTYPE_t augment_amount = round(score - running_total)

    return score, included_i.astype(int), included_j.astype(int), included_label.astype(int), included_father.astype(int), included_type.astype(int), augment_amount
