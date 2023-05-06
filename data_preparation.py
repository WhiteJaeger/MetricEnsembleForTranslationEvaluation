import csv
import random

DATA = []


def collect_comet_and_add_to_data(path_to_reference_1_file: str,
                                  path_to_reference_2_file: str,
                                  path_to_comet_scores: str) -> None:
    with open(path_to_comet_scores, 'r') as scores_file, \
            open(path_to_reference_1_file) as reference_1_file, \
            open(path_to_reference_2_file) as reference_2_file:
        scores = []

        reader = csv.reader(scores_file, delimiter='\t')
        for line in reader:
            scores.append(float(line[2]))

        references_1, references_2 = reference_1_file.read().split('\n')[:len(scores)], \
            reference_2_file.read(
            ).split('\n')[:len(scores)]

        for sentence_1, sentence_2, score in zip(references_1, references_2, scores):
            if sentence_1 == sentence_2:
                score = 1
            DATA.append([sentence_1, sentence_2, score])


collect_comet_and_add_to_data(path_to_reference_1_file='/Users/andrej/PycharmProjects'
                                                       '/NNforTranslationEvaluation/data/comet'
                                                       '/wmt20_reference_1.txt',
                              path_to_reference_2_file='/Users/andrej/PycharmProjects'
                                                       '/NNforTranslationEvaluation/data/comet'
                                                       '/wmt20_reference_2.txt',
                              path_to_comet_scores='/Users/andrej/PycharmProjects'
                                                   '/NNforTranslationEvaluation/data/comet'
                                                   '/comet_scores/wmt20_ru_en.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt21.news'
                             '/references/ru-en.refA'
                             '.txt',
    path_to_reference_2_file='/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt21.news'
                             '/references/ru-en.refB.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores'
                         '/wmt21news_ru_en.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system'
                             '-outputs/ru-en/PROMT.txt',
    path_to_reference_2_file='/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system'
                             '-outputs/ru-en/refA.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores'
                         '/wmt22_promt_reference.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system'
                             '-outputs/ru-en/PROMT.txt',
    path_to_reference_2_file='/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system'
                             '-outputs/ru-en/HuaweiTSC.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores'
                         '/wmt22_promt_huawei.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system'
                             '-outputs/ru-en/PROMT.txt',
    path_to_reference_2_file='/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system'
                             '-outputs/ru-en/SRPOL.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores'
                         '/wmt22_promt_srpol.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt20/references'
                             '/ru-en.ref.txt',
    path_to_reference_2_file='/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system'
                             '-outputs/ru-en/refA.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores/wmt20_ref_wmt22_ref_ru_en.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt20/system-outputs/ru-en/ref.txt',
    path_to_reference_2_file='/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system-outputs/ru-en/refA.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores/wmt20_mt_ref_wmt22_mt_ref.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt21.tedtalks/system-outputs/zh-en/Facebook-AI.txt',
    path_to_reference_2_file='/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system-outputs/ru-en/refA.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores/wmt21ted_zh_en_facebook_wmt22_mt_ref.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/german_baseline_reference.txt',
    path_to_reference_2_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/german_baseline_mt.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores/demetr_ge_mt_reference.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/german_shuffled_perturbation.txt',
    path_to_reference_2_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/german_baseline_reference.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores/demetr_ge_shuffled_reference.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/russian_baseline_reference.txt',
    path_to_reference_2_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/russian_baseline_mt.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores/demetr_ru_mt_reference.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/russian_shuffled_perturbation.txt',
    path_to_reference_2_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/russian_baseline_reference.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores/demetr_ru_shuffled_reference.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/french_baseline_reference.txt',
    path_to_reference_2_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/russian_baseline_reference.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores/demetr_ru_source_fr_reference_ru_reference.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/german_baseline_reference.txt',
    path_to_reference_2_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/russian_baseline_reference.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores/demetr_ru_source_ge_reference_ru_reference.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/spanish_baseline_reference.txt',
    path_to_reference_2_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/russian_baseline_reference.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores/demetr_ru_source_sp_reference_ru_reference.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/spanish_baseline_reference.txt',
    path_to_reference_2_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/spanish_baseline_mt.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores/demetr_sp_mt_reference.txt')

collect_comet_and_add_to_data(
    path_to_reference_1_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/spanish_shuffled_perturbation.txt',
    path_to_reference_2_file='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                             '/spanish_baseline_reference.txt',
    path_to_comet_scores='/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
                         '/comet_scores/demetr_sp_shuffled_reference.txt')

with open('/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/wiki_sentences.txt') as \
        fd, \
        open('/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/Translator1.txt') \
                as fd2:
    _data = fd.read().split('\n')
    _data2 = fd2.read().split('\n')
    _data = random.sample(_data, len(_data2))

    for sent_1, sent_2 in zip(_data, _data2):
        if sent_1 and sent_2:
            DATA.append([sent_1, sent_2, 0])

with open('/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/wiki_sentences.txt') as \
        fd, \
        open('/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/Translator2.txt') \
                as fd2:
    _data = fd.read().split('\n')
    _data2 = fd2.read().split('\n')
    _data = random.sample(_data, len(_data2))

    for sent_1, sent_2 in zip(_data, _data2):
        if sent_1 and sent_2:
            DATA.append([sent_1, sent_2, 0])

with open('/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/wiki_sentences.txt') as \
        fd, \
        open('/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/Translator3.txt') \
                as fd2:
    _data = fd.read().split('\n')
    _data2 = fd2.read().split('\n')
    _data = random.sample(_data, len(_data2))

    for sent_1, sent_2 in zip(_data, _data2):
        if sent_1 and sent_2:
            DATA.append([sent_1, sent_2, 0])

with open('/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/wiki_sentences.txt') as \
        fd, \
        open('/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/Translator4.txt') \
                as fd2:
    _data = fd.read().split('\n')
    _data2 = fd2.read().split('\n')
    _data = random.sample(_data, len(_data2))

    for sent_1, sent_2 in zip(_data, _data2):
        if sent_1 and sent_2:
            DATA.append([sent_1, sent_2, 0])

with open('/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/wiki_sentences.txt') as \
        fd:
    lines = fd.read().split('\n')
    _data = lines[:2000]
    _data2 = lines[2001:]
    _data = random.sample(_data, 1700)
    _data2 = random.sample(_data2, 1700)

    for sent_1, sent_2 in zip(_data, _data2):
        if sent_1 and sent_2:
            DATA.append([sent_1, sent_2, 0])

print(f'\nDATA LENGTH: {len(DATA)}\n')
