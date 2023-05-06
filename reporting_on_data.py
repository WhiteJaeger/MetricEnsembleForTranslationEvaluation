from nltk.tokenize import word_tokenize


paths_wmt = ['/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt21.news/references/ru-en.refB.txt',
             '/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system-outputs/ru-en/SRPOL.txt',
             '/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt21.tedtalks/system-outputs/zh-en/Facebook-AI.txt',
             '/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system-outputs/ru-en/PROMT.txt',
             '/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt21.news/references/ru-en.refA.txt',
             '/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt20/references/ru-en.ref.txt',
             '/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system-outputs/ru-en/HuaweiTSC.txt',
             '/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt20/system-outputs/ru-en/ref.txt',
             '/Users/andrej/.mt-metrics-eval/mt-metrics-eval-v2/wmt22/system-outputs/ru-en/refA.txt']

paths_wiki = ['/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/wiki_sentences.txt']

paths_demetr = [
    '/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet/german_baseline_reference.txt',
    '/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet/spanish_shuffled_perturbation.txt',
    '/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet/french_baseline_reference.txt',
    '/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet/russian_baseline_mt.txt',
    '/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet/german_baseline_mt.txt',
    '/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet/russian_shuffled_perturbation.txt',
    '/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet/german_shuffled_perturbation.txt',
    '/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet/spanish_baseline_reference.txt',
    '/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet/russian_baseline_reference.txt',
    '/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet/spanish_baseline_mt.txt',
]

path_by_group = {
    'DEMETR': paths_demetr,
    'WMT': paths_wmt,
    'WIKI': paths_wiki
}


total_characters = 0
total_tokens = 0
total_sentences = 0

for path_group, paths in path_by_group.items():

    total_characters_per_group = 0
    total_tokens_per_group = 0
    total_sentences_per_group = 0

    for path in paths:
        with open(path, 'r', encoding='utf-8') as fd:
            data = fd.read().split('\n')

            total_sentences += len(data)
            total_sentences_per_group += len(data)

            for sentence in data:
                words = word_tokenize(sentence)

                total_tokens += len(words)
                total_tokens_per_group += len(words)

                total_characters += sum([len(word) for word in words])
                total_characters_per_group += sum([len(word) for word in words])

    print(f'\nGROUP: {path_group}\nTOTAL CHARACTERS: {total_characters_per_group}\nTOTAL TOKENS: {total_tokens_per_group}\nTOTAL SENTENCES: {total_sentences_per_group}\n')

print(f'ACROSS ALL.\nTOTAL CHARACTERS: {total_characters}\nTOTAL TOKENS: {total_tokens}\nTOTAL SENTENCES: {total_sentences}')
