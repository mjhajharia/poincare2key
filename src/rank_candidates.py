from utils import texttofreq
import statistics
import math


def rank(ps, text, candidates):
    for punc in list("""!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~"""):
        text = text.replace(punc, " ")

    text = text.lower()
    freq = texttofreq(ps, text)

    score = {}

    for candidate in candidates:

        if candidate == "":
            continue
        token_score = text.count(candidate)

        tokens = candidate.split()
        stemmed_cand = []
        score_list = []

        for i, token in enumerate(tokens):
            score_list.append(freq.get(ps.stem(token), 0))
            stemmed_cand.append(ps.stem(token))

        if len(score_list) == 1:
            sum_token_score = 1
        else:
            if len(score_list) % 2:
                sum_token_score = statistics.median(score_list)
            else:
                sum_token_score = sorted(score_list, reverse=True)[math.ceil(len(score_list) / 2)]

        score[" ".join(stemmed_cand)] = token_score * sum_token_score

    ranked_candidates = [(k, v) for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)]
    ranked_candidates = rank_by_pos(ps, text, ranked_candidates)
    ranked_candidates = remove_dup_unigram(ranked_candidates)

    return ranked_candidates


def rank_by_pos(ps, text, ranked_candidates):
    text = text.lower().split()
    text = [ps.stem(word) for word in text]
    text = " ".join(text)

    bucket = {}
    for key in ranked_candidates:
        if key[0] in text:
            pos = text.index(key[0]) / len(text)
        else:
            pos = 1
        if key[1] in bucket:
            bucket[key[1]].append((key[0], pos / len(key[0])))
        else:
            bucket[key[1]] = [(key[0], pos)]
    final_rank = []
    for score in bucket:
        if len(bucket[score]) == 1:
            final_rank.append(bucket[score][0][0])
        else:
            new = [x[0] for x in sorted(bucket[score], key=lambda x: x[1])]
            final_rank.extend(new)

    return final_rank


def remove_dup_unigram(ranked_candidates):
    final_ranked = []
    i = 0
    while i < len(ranked_candidates):
        j = 0
        found = False
        while j < i:
            a = ranked_candidates[i].split()
            b = ranked_candidates[j].split()
            c = set(a).intersection(set(b))
            if len(c) / max(len(a), len(b)) >= 0.65:
                found = True
            j += 1

        if not found:
            final_ranked.append(ranked_candidates[i])
        i += 1

    return final_ranked
