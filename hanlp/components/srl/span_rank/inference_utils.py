# Adopted from https://github.com/KiroSummer/A_Syntax-aware_MTL_Framework_for_Chinese_SRL

# Inference functions for the SRL model.
import numpy as np


def decode_spans(span_starts, span_ends, span_scores, labels_inv):
    """

    Args:
      span_starts: [num_candidates,]
      span_scores: [num_candidates, num_labels]
      span_ends: 
      labels_inv: 

    Returns:

    
    """
    pred_spans = []
    span_labels = np.argmax(span_scores, axis=1)  # [num_candidates]
    spans_list = list(zip(span_starts, span_ends, span_labels, span_scores))
    spans_list = sorted(spans_list, key=lambda x: x[3][x[2]], reverse=True)
    predicted_spans = {}
    for start, end, label, _ in spans_list:
        # Skip invalid span.
        if label == 0 or (start, end) in predicted_spans:
            continue
        pred_spans.append((start, end, labels_inv[label]))
        predicted_spans[(start, end)] = label
    return pred_spans


def greedy_decode(predict_dict, srl_labels_inv):
    """Greedy decoding for SRL predicate-argument structures.

    Args:
      predict_dict: Dictionary of name to numpy arrays.
      srl_labels_inv: SRL label id to string name.
      suppress_overlap: Whether to greedily suppress overlapping arguments for the same predicate.

    Returns:

    
    """
    arg_starts = predict_dict["arg_starts"]
    arg_ends = predict_dict["arg_ends"]
    predicates = predict_dict["predicates"]
    arg_labels = predict_dict["arg_labels"]
    scores = predict_dict["srl_scores"]

    num_suppressed_args = 0

    # Map from predicates to a list of labeled spans.
    pred_to_args = {}
    if len(arg_ends) > 0 and len(predicates) > 0:
        max_len = max(np.max(arg_ends), np.max(predicates)) + 1
    else:
        max_len = 1

    for j, pred_id in enumerate(predicates):
        args_list = []
        for i, (arg_start, arg_end) in enumerate(zip(arg_starts, arg_ends)):
            # If label is not null.
            if arg_labels[i][j] == 0:
                continue
            label = srl_labels_inv[arg_labels[i][j]]
            # if label not in ["V", "C-V"]:
            args_list.append((arg_start, arg_end, label, scores[i][j][arg_labels[i][j]]))

        # Sort arguments by highest score first.
        args_list = sorted(args_list, key=lambda x: x[3], reverse=True)
        new_args_list = []

        flags = [False for _ in range(max_len)]
        # Predicate will not overlap with arguments either.
        flags[pred_id] = True

        for (arg_start, arg_end, label, score) in args_list:
            # If none of the tokens has been covered:
            if not max(flags[arg_start:arg_end + 1]):
                new_args_list.append((arg_start, arg_end, label))
                for k in range(arg_start, arg_end + 1):
                    flags[k] = True

        # Only add predicate if it has any argument.
        if new_args_list:
            pred_to_args[pred_id] = new_args_list

        num_suppressed_args += len(args_list) - len(new_args_list)

    return pred_to_args, num_suppressed_args


_CORE_ARGS = {"ARG0": 1, "ARG1": 2, "ARG2": 4, "ARG3": 8, "ARG4": 16, "ARG5": 32, "ARGA": 64,
              "A0": 1, "A1": 2, "A2": 4, "A3": 8, "A4": 16, "A5": 32, "AA": 64}


def get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents):
    mention_to_predicted = {}
    predicted_clusters = []
    for i, predicted_index in enumerate(predicted_antecedents):
        if predicted_index < 0:
            continue
        assert i > predicted_index
        predicted_antecedent = (int(top_span_starts[predicted_index]), int(top_span_ends[predicted_index]))
        if predicted_antecedent in mention_to_predicted:
            predicted_cluster = mention_to_predicted[predicted_antecedent]
        else:
            predicted_cluster = len(predicted_clusters)
            predicted_clusters.append([predicted_antecedent])
            mention_to_predicted[predicted_antecedent] = predicted_cluster

        mention = (int(top_span_starts[i]), int(top_span_ends[i]))
        predicted_clusters[predicted_cluster].append(mention)
        mention_to_predicted[mention] = predicted_cluster

    predicted_clusters = [tuple(pc) for pc in predicted_clusters]
    mention_to_predicted = {m: predicted_clusters[i] for m, i in list(mention_to_predicted.items())}

    return predicted_clusters, mention_to_predicted


def _decode_non_overlapping_spans(starts, ends, scores, max_len, labels_inv, pred_id):
    labels = np.argmax(scores, axis=1)
    spans = []
    for i, (start, end, label) in enumerate(zip(starts, ends, labels)):
        if label <= 0:
            continue
        label_str = labels_inv[label]
        if pred_id is not None and label_str == "V":
            continue
        spans.append((start, end, label_str, scores[i][label]))
    spans = sorted(spans, key=lambda x: x[3], reverse=True)
    flags = np.zeros([max_len], dtype=bool)
    if pred_id is not None:
        flags[pred_id] = True
    new_spans = []
    for start, end, label_str, score in spans:
        if not max(flags[start:end + 1]):
            new_spans.append((start, end, label_str))  # , score))
            for k in range(start, end + 1):
                flags[k] = True
    return new_spans


def _dp_decode_non_overlapping_spans(starts, ends, scores, max_len, labels_inv, pred_id, u_constraint=False):
    num_roles = scores.shape[1]  # [num_arg, num_roles]
    labels = np.argmax(scores, axis=1).astype(np.int64)
    spans = list(zip(starts, ends, list(range(len(starts)))))
    spans = sorted(spans, key=lambda x: (x[0], x[1]))  # sort according to the span start index

    if u_constraint:
        f = np.zeros([max_len + 1, 128], dtype=float) - 0.1
    else:  # This one
        f = np.zeros([max_len + 1, 1], dtype=float) - 0.1

    f[0, 0] = 0
    states = {0: set([0])}  # A dictionary from id to list of binary core-arg states.
    pointers = {}  # A dictionary from states to (arg_id, role, prev_t, prev_rs)
    best_state = [(0, 0)]

    def _update_state(t0, rs0, t1, rs1, delta, arg_id, role):
        if f[t0][rs0] + delta > f[t1][rs1]:
            f[t1][rs1] = f[t0][rs0] + delta
            if t1 not in states:
                states[t1] = set()
            states[t1].update([rs1])
            pointers[(t1, rs1)] = (arg_id, role, t0, rs0)  # the pointers store
            if f[t1][rs1] > f[best_state[0][0]][best_state[0][1]]:
                best_state[0] = (t1, rs1)

    for start, end, i in spans:  # [arg_start, arg_end, arg_span_id]
        assert scores[i][0] == 0  # dummy score
        # The extra dummy score should be same for all states, so we can safely skip arguments overlap
        # with the predicate.
        if pred_id is not None and start <= pred_id and pred_id <= end:  # skip the span contains the predicate
            continue
        r0 = labels[i]  # Locally best role assignment.
        # Strictly better to incorporate a dummy span if it has the highest local score.
        if r0 == 0:  # labels_inv[r0] == "O"
            continue
        r0_str = labels_inv[r0]
        # Enumerate explored states.
        t_states = [t for t in list(states.keys()) if t <= start]  # collect the state which is before the current span
        for t in t_states:  # for each state
            role_states = states[t]
            # Update states if best role is not a core arg.
            if not u_constraint or r0_str not in _CORE_ARGS:  # True; this one
                for rs in role_states:  # the set type in the value in the state dict
                    _update_state(t, rs, end + 1, rs, scores[i][r0], i, r0)  # update the state
            else:
                for rs in role_states:
                    for r in range(1, num_roles):
                        if scores[i][r] > 0:
                            r_str = labels_inv[r]
                            core_state = _CORE_ARGS.get(r_str, 0)
                            # print start, end, i, r_str, core_state, rs
                            if core_state & rs == 0:
                                _update_state(t, rs, end + 1, rs | core_state, scores[i][r], i, r)
    # Backtrack to decode.
    new_spans = []
    t, rs = best_state[0]
    while (t, rs) in pointers:
        i, r, t0, rs0 = pointers[(t, rs)]
        new_spans.append((int(starts[i]), int(ends[i]), labels_inv[r]))
        t = t0
        rs = rs0
    return new_spans[::-1]


def srl_decode(sentence_lengths, predict_dict, srl_labels_inv, config):  # decode the predictions.
    # Decode sentence-level tasks.
    num_sentences = len(sentence_lengths)
    predictions = [{} for _ in range(num_sentences)]
    # Sentence-level predictions.
    for i in range(num_sentences):  # for each sentences
        # if predict_dict["No_arg"] is True:
        #     predictions["srl"][i][predict_dict["predicates"][i]] = []
        #     continue
        predict_dict_num_args_ = predict_dict["num_args"].cpu().numpy()
        predict_dict_num_preds_ = predict_dict["num_preds"].cpu().numpy()
        predict_dict_predicates_ = predict_dict["predicates"].cpu().numpy()
        predict_dict_arg_starts_ = predict_dict["arg_starts"].cpu().numpy()
        predict_dict_arg_ends_ = predict_dict["arg_ends"].cpu().numpy()
        predict_dict_srl_scores_ = predict_dict["srl_scores"].detach().cpu().numpy()
        num_args = predict_dict_num_args_[i]  # the number of the candidate argument spans
        num_preds = predict_dict_num_preds_[i]  # the number of the candidate predicates
        # for each predicate id, exec the decode process
        for j, pred_id in enumerate(predict_dict_predicates_[i][:num_preds]):
            # sorted arg_starts and arg_ends and srl_scores ? should be??? enforce_srl_constraint = False
            arg_spans = _dp_decode_non_overlapping_spans(
                predict_dict_arg_starts_[i][:num_args],
                predict_dict_arg_ends_[i][:num_args],
                predict_dict_srl_scores_[i, :num_args, j, :],
                sentence_lengths[i], srl_labels_inv, pred_id, config.enforce_srl_constraint)
            # To avoid warnings in the eval script.
            if config.use_gold_predicates:  # false
                arg_spans.append((pred_id, pred_id, "V"))
            if arg_spans:
                predictions[i][int(pred_id)] = sorted(arg_spans, key=lambda x: (x[0], x[1]))

    return predictions
