from pathlib import Path

import penman


def write_predictions(predictions_path, tokenizer, graphs):
    pieces = [penman.encode(g) for g in graphs]
    text = '\n\n'.join(pieces)
    if tokenizer:
        text = text.replace(tokenizer.INIT, '')
    Path(predictions_path).write_text(text)
    return predictions_path


def compute_smatch(pred, gold):
    from perin_parser.thirdparty.mtool import smatch
    with Path(pred).open() as p, Path(gold).open() as g:
        score = next(smatch.score_amr_pairs(p, g))
    return score[2]


def compute_bleu(gold_sentences, pred_sentences):
    from sacrebleu import corpus_bleu
    return corpus_bleu(pred_sentences, [gold_sentences])
