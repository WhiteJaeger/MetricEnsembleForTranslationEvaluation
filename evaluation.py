from pathlib import Path

import spacy
import torch
from mt_metrics_eval import data
from nltk.tokenize import word_tokenize
from nltk.translate import bleu_score, meteor_score, chrf_score
from subtree_metric import stm

from nn_training import MyNetwork, POS_WEIGHTS

spacy_model = spacy.load('en_core_web_md')

checkpoint = None
model_path = Path(__file__).parent.joinpath('models'). \
    joinpath(f'metric_ensemble_{checkpoint}.pt')

net: MyNetwork = torch.load(model_path)


def metric_wrapper_nn(hypothesis: str, reference: str):
    # seg: Pearson=0.188439, Kendall-like=0.090274
    return float(net(hypothesis, reference)[0])


def metric_wrapper_stm(hypothesis: str, reference: str):
    # seg: Pearson=0.121566, Kendall-like=-0.097975
    return stm.sentence_stm(reference,
                            hypothesis,
                            spacy_model)


def metric_wrapper_stm_augmented(hypothesis: str, reference: str):
    # seg: Pearson=0.138049, Kendall-like=-0.032088
    return stm.sentence_stm_augmented(reference=reference,
                                      hypothesis=hypothesis,
                                      nlp_model=spacy_model,
                                      pos_weights=POS_WEIGHTS,
                                      depth=3)


def metric_wrapper_bleu(hypothesis: str, reference: str):
    # seg: Pearson=0.144735, Kendall-like=-0.000570
    return bleu_score.sentence_bleu([word_tokenize(reference)], word_tokenize(hypothesis))


def metric_wrapper_meteor(hypothesis: str, reference: str):
    # seg: Pearson=0.189121, Kendall-like=-0.001854
    return meteor_score.single_meteor_score(word_tokenize(reference), word_tokenize(hypothesis))


def metric_wrapper_chrf(hypothesis: str, reference: str):
    # seg: Pearson=0.191591, Kendall-like=0.048203
    return chrf_score.sentence_chrf(reference.split(), hypothesis.split())


evs = data.EvalSet('wmt20', 'ru-en')
scores = {level: {} for level in ['seg']}
references = evs.all_refs[evs.std_ref]
for s, hypotheses in evs.sys_outputs.items():
    scores['seg'][s] = [metric_wrapper_stm_augmented(h, r) for h, r in zip(hypotheses, references)]

# Official WMT correlations
for level in ['seg']:
    gold_scores = evs.Scores(level, evs.StdHumanScoreName(level))
    sys_names = set(gold_scores) - evs.human_sys_names
    corr = evs.Correlation(gold_scores, scores[level], sys_names)
    print(f'{level}: Pearson={corr.Pearson()[0]:f}, '
          f'Kendall-like={corr.KendallLike()[0]:f}')
