from pathlib import Path

import spacy
import torch
from mt_metrics_eval import data
from nltk.tokenize import word_tokenize
from nltk.translate import bleu_score, meteor_score, chrf_score
from subtree_metric import stm

from training_conv import EMEQT, POS_WEIGHTS
from typing import Callable

spacy_model = spacy.load('en_core_web_md')

model_name = 'metric_ensemble_conv_2023-05-05_23:53:42.pt'
model_path = Path(__file__).parent.joinpath('models'). \
    joinpath(model_name)

net: EMEQT = torch.load(model_path)


def metric_wrapper_nn(hypothesis: str, reference: str):
    # wmt19 - ru-en Pearson=0.243181, Kendall-like=0.157411
    # wmt20 - ru-en Pearson=0.197218, Kendall-like=0.090274
    # wmt21.news - ru-en Pearson=-0.018269, Kendall-like=-0.113347
    # wmt22 - ru-en Pearson=0.021965, Kendall-like=-0.151331
    return float(net(hypothesis, reference)[0])


def metric_wrapper_stm(hypothesis: str, reference: str):
    # wmt19 - ru-en Pearson=0.176296, Kendall-like=0.016070
    # wmt20 - ru-en Pearson=0.121566, Kendall-like=-0.097975
    # wmt21.news - ru-en Pearson=-0.007766, Kendall-like=-0.195874
    # wmt22 - ru-en Pearson=0.018541, Kendall-like=-0.255816
    return stm.sentence_stm(reference,
                            hypothesis,
                            spacy_model)


def metric_wrapper_stm_augmented(hypothesis: str, reference: str):
    # wmt19 - ru-en Pearson=0.196198, Kendall-like=0.068786
    # wmt20 - ru-en Pearson=0.138049, Kendall-like=-0.032088
    # wmt21.news - ru-en Pearson=-0.014008, Kendall-like=-0.142511
    # wmt22 - ru-en Pearson=0.017211, Kendall-like=-0.200975
    return stm.sentence_stm_augmented(reference=reference,
                                      hypothesis=hypothesis,
                                      nlp_model=spacy_model,
                                      pos_weights=POS_WEIGHTS,
                                      depth=3)


def metric_wrapper_bleu(hypothesis: str, reference: str):
    # wmt19 - ru-en Pearson=0.212407, Kendall-like=0.124361
    # wmt20 - ru-en Pearson=0.144735, Kendall-like=-0.000570
    # wmt21.news - ru-en Pearson=-0.018255, Kendall-like=-0.146460
    # wmt22 - ru-en Pearson=0.017087, Kendall-like=-0.185307
    return bleu_score.sentence_bleu([word_tokenize(reference)], word_tokenize(hypothesis))


def metric_wrapper_meteor(hypothesis: str, reference: str):
    # wmt19 - ru-en Pearson=0.264139, Kendall-like=0.132201
    # wmt20 - ru-en Pearson=0.189121, Kendall-like=-0.001854
    # wmt21.news - ru-en Pearson=-0.023870, Kendall-like=-0.162252
    # wmt22 - ru-en Pearson=0.027257, Kendall-like=-0.183308
    return meteor_score.single_meteor_score(word_tokenize(reference), word_tokenize(hypothesis))


def metric_wrapper_chrf(hypothesis: str, reference: str):
    # wmt19 - ru-en Pearson=0.289186, Kendall-like=0.180282
    # wmt20 - ru-en Pearson=0.191591, Kendall-like=0.048203
    # wmt21.news - ru-en Pearson=-0.018956, Kendall-like=-0.128884
    # wmt22 - ru-en Pearson=0.018816, Kendall-like=-0.148933
    return chrf_score.sentence_chrf(reference.split(), hypothesis.split())


def compute_and_display_correlation_for_metric(metric_wrapper: Callable,
                                               metric_name: str,
                                               dataset_name: str,
                                               dataset_language_pair: str) -> None:
    level = 'seg'
    evs = data.EvalSet(dataset_name, dataset_language_pair)
    scores = {level: {} for level in ['seg']}
    references = evs.all_refs[evs.std_ref]
    for s, hypotheses in evs.sys_outputs.items():
        scores['seg'][s] = [metric_wrapper(h, r) for h, r in zip(hypotheses, references)]

    # Official WMT correlations
    gold_scores = evs.Scores(level, evs.StdHumanScoreName(level))
    sys_names = set(gold_scores) - evs.human_sys_names
    corr = evs.Correlation(gold_scores, scores[level], sys_names)
    print(f'METRIC NAME: {metric_name}. DATA: {dataset_name} - {dataset_language_pair} '
          f'Pearson={corr.Pearson()[0]:f}, Kendall-like={corr.KendallLike()[0]:f}')


metrics = (('BLEU', metric_wrapper_bleu), ('CHRF', metric_wrapper_chrf),
           ('METEOR', metric_wrapper_meteor), ('STM', metric_wrapper_stm),
           ('STM-A', metric_wrapper_stm_augmented), ('EMEQT', metric_wrapper_nn))


datasets = ('wmt19', 'wmt20', 'wmt21.news', 'wmt22')

for metric in metrics:
    name, wrapper = metric
    compute_and_display_correlation_for_metric(wrapper,
                                               name,
                                               'wmt21.news',
                                               'ru-en')
