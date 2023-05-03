import json
from pathlib import Path


def prepare_from_json(path_to_json: Path) -> None:
    data_folder = path_to_json.parent
    _type = path_to_json.stem.split('_')[-1]

    with open(path_to_json,
              'r') as fd:
        raw_data = json.load(fd)

        _data = {}

        for line in raw_data:
            if line['lang_tag'] in _data:
                continue
            else:
                _data[line['lang_tag']] = {
                    'sources': [],
                    'machine_translations': [],
                    'references': [],
                    'perturbations': []
                }

        for line in raw_data:
            _data[line['lang_tag']]['sources'].append(line['src_sent'])
            _data[line['lang_tag']]['machine_translations'].append(line['mt_sent'])
            _data[line['lang_tag']]['references'].append(line['eng_sent'])
            _data[line['lang_tag']]['perturbations'].append(line['pert_sent'])

    for language in _data.keys():
        with open(data_folder.joinpath(f'{language}_{_type}_source.txt'), 'w',
                  encoding='utf-8') as fd:
            fd.write('\n'.join(_data[language]['sources']))

        with open(data_folder.joinpath(f'{language}_{_type}_reference.txt'), 'w',
                  encoding='utf-8') as fd:
            fd.write('\n'.join(_data[language]['references']))

        with open(data_folder.joinpath(f'{language}_{_type}_mt.txt'), 'w',
                  encoding='utf-8') as fd:
            fd.write('\n'.join(_data[language]['machine_translations']))

        with open(data_folder.joinpath(f'{language}_{_type}_perturbation.txt'), 'w',
                  encoding='utf-8') as fd:
            fd.write('\n'.join(_data[language]['perturbations']))


paths = ['/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet/multi_antonym.json',
         '/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet/multi_baseline.json',
         '/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet/multi_removed.json',
         '/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet/multi_shuffled.json']
paths = list(map(Path, paths))

for path in paths:
    prepare_from_json(path)

with open('/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
          '/data_wmt20_with_src.txt', encoding='utf-8') as fd:
    lines = fd.read().split('\n')
    sources = []
    reference_1 = []
    reference_2 = []
    for line in lines:
        try:
            sources.append(line.split('@@@')[0])
            reference_1.append(line.split('@@@')[1])
            reference_2.append(line.split('@@@')[2])
        except:
            pass
    with open('/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet/wmt20_src.txt', 'w') as _fd:
        _fd.write('\n'.join(sources))

    with open('/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
              '/wmt20_reference_1'
              '.txt'
              '', 'w') as _fd:
        _fd.write('\n'.join(reference_1))

    with open('/Users/andrej/PycharmProjects/NNforTranslationEvaluation/data/comet'
              '/wmt20_reference_2'
              '.txt', 'w') as _fd:
        _fd.write('\n'.join(reference_2))
