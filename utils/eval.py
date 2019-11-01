import sys
from collections import defaultdict

from Levenshtein import distance
import pandas as pd


def load_data(predicted_path, gold_path):
    with open(predicted_path, 'r', encoding='utf-8') as file:
        predicted_text = file.readlines()
        if len(predicted_text[0].split()[0]) == 3:
            # use this line if we included the tags in the predicted file
            predicted_text = [p.rstrip('\n')[4:].replace(' ', '') for p in predicted_text]
        else:
            # use this line if we didn't include the tags in the predicted file
            predicted_text = [p.rstrip('\n').replace(' ', '') for p in predicted_text]
    with open(gold_path, 'r', encoding='utf-8') as file:
        gold_text = file.readlines()
        gold_text = [g.rstrip('\n').strip()[4:].replace(' ', '') for g in gold_text]
    return predicted_text, gold_text


def load_data_by_language(predicted_path, gold_path):
    predicted_text_by_language = defaultdict(list)
    gold_text_by_language = defaultdict(list)
    with open(predicted_path, 'r', encoding='utf-8') as file:
        predicted_text = file.readlines()
        if len(predicted_text[0].split()[0]) == 3:
            # use this line if we included the tags in the predicted file
            predicted_text = [p.rstrip('\n')[4:].replace(' ', '') for p in predicted_text]
        else:
            # use this line if we didn't include the tags in the predicted file
            predicted_text = [p.rstrip('\n').replace(' ', '') for p in predicted_text]
    with open(gold_path, 'r', encoding='utf-8') as file:
        gold_text = file.readlines()
        gold_text = [g.rstrip('\n').strip() for g in gold_text]
    for predicted, gold in zip(predicted_text, gold_text):
        predicted_text_by_language[gold[:3]].append(predicted)
        gold_text_by_language[gold[:3]].append(gold[4:].replace(' ', ''))

    return predicted_text_by_language, gold_text_by_language


def levenshtein(a, b):
    return distance(a, b)


# for comparison: from the baseline paper's repo. Slower, but same results as Levenshtein.distance()
# def levenshtein(a, b):
# 	d = [[0 for i in range(len(b) + 1)] for j in range(len(a) + 1)]
# 	for i in range(1, len(a) + 1):
# 		d[i][0] = i
# 	for j in range(1, len(b) + 1):
# 		d[0][j] = j
# 	for j in range(1, len(b) + 1):
# 		for i in range(1, len(a) + 1):
# 			cost = int(a[i - 1] != b[j - 1])
# 			d[i][j] = min(d[i][j - 1] + 1, d[i - 1][j] + 1, d[i - 1][j - 1] + cost)
# 	return d[len(a)][len(b)]


def calc_word_error_rate(predicted_text, gold_text):
    num_errors = sum(p != g for p, g in zip(predicted_text, gold_text))
    num_elements_pred = len(predicted_text)
    return num_errors / num_elements_pred


def calc_phoneme_error_rate(predicted_text, gold_text, quiet=True):
    total_distance = sum(levenshtein(p, g) for p, g in zip(predicted_text, gold_text))
    num_elements_gold = sum(len(g) for g in gold_text)
    num_elements_pred = sum(len(p) for p in predicted_text)
    if not quiet:
        print("Total edit distance: ", total_distance)
        print("Number of characters in gold text: ", num_elements_gold)
        print("Number of character in predicted text: ", num_elements_pred)
    return total_distance / num_elements_gold


def main():
    predicted_path = sys.argv[1]
    gold_path = sys.argv[2]
    predicted_text, gold_text = load_data(predicted_path, gold_path)

    phoneme_error_rate = calc_phoneme_error_rate(predicted_text, gold_text)
    word_error_rate = calc_word_error_rate(predicted_text, gold_text)
    print("Phoneme error rate is {:0.4f}".format(phoneme_error_rate))
    print("Word error rate is {:0.4f}".format(word_error_rate))

    predicted_text_by_language, gold_text_by_language = load_data_by_language(predicted_path, gold_path)
    phoneme_and_word_error_rate_by_language = dict()
    for language in predicted_text_by_language.keys():
        phoneme_error_rate = "{:0.4f}".format(calc_phoneme_error_rate(predicted_text_by_language[language], gold_text_by_language[language]))
        word_error_rate = "{:0.4f}".format(calc_word_error_rate(predicted_text_by_language[language], gold_text_by_language[language]))
        phoneme_and_word_error_rate_by_language[language] = (phoneme_error_rate, word_error_rate)
    pd.DataFrame.from_dict(phoneme_and_word_error_rate_by_language, orient='index', columns=['PER', 'WER']).to_csv(predicted_path.split('/')[1].split('.')[0] + '_scores_by_language.csv')


if __name__ == '__main__':
    main()
