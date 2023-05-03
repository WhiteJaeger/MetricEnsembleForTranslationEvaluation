import spacy
from mt_metrics_eval import data
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.chrf_score import sentence_chrf
from nltk.translate.meteor_score import single_meteor_score
from subtree_metric import stm

spacy_model: spacy.Language = spacy.load('en_core_web_md')


def stm_wrapper(out, ref):
    return stm.sentence_stm(reference=ref,
                            hypothesis=out,
                            nlp_model=spacy_model)


def bleu_wrapper(out, ref):
    return sentence_bleu([ref], out)


def meteor_wrapper(out, ref):
    return single_meteor_score(word_tokenize(ref), word_tokenize(out))


def chrf_wrapper(out, ref):
    return sentence_chrf(word_tokenize(ref), word_tokenize(out))


evs = data.EvalSet('wmt20', 'ru-en')
scores = {level: {} for level in ['seg']}
ref = evs.all_refs[evs.std_ref]
for s, out in evs.sys_outputs.items():
    scores['seg'][s] = [chrf_wrapper(o, r) for o, r in zip(out, ref)]

# Official WMT correlations.
gold_scores = evs.Scores('seg', evs.StdHumanScoreName('seg'))
print(gold_scores)
sys_names = set(gold_scores) - evs.human_sys_names
corr = evs.Correlation(gold_scores, scores['seg'], sys_names)
print(f'{"seg"}: Pearson={corr.Pearson()[0]:f}, '
      f'Kendall-like={corr.KendallLike()[0]:f}')
