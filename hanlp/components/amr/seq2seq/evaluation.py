from pathlib import Path

import penman


def write_predictions(predictions_path, tokenizer, graphs):
    pieces = [penman.encode(g) for g in graphs]
    Path(predictions_path).write_text('\n\n'.join(pieces).replace(tokenizer.INIT, ''))
    return predictions_path


def compute_smatch(pred, gold):
    import smatch
    with Path(pred).open() as p, Path(gold).open() as g:
        score = next(smatch.score_amr_pairs(p, g))
    return score[2]


def compute_bleu(gold_sentences, pred_sentences):
    from sacrebleu import corpus_bleu
    return corpus_bleu(pred_sentences, [gold_sentences])
