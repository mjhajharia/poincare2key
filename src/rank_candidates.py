from utils import texttofreq


def rank(ps, text, candidates):
    for punc in list("""!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~"""):
        text = text.replace(punc, " ")
    freq = texttofreq(ps, text)

    score = {}

    for candidate in candidates:
        tokens = candidate.split()
        token_score = 0
        stemmed_cand = []
        for token in tokens:
            token_score += freq.get(ps.stem(token), 0)
            stemmed_cand.append(ps.stem(token))

        score[" ".join(stemmed_cand)] = token_score

    ranked_candidates = [k for k, v in sorted(score.items(), key=lambda item: item[1], reverse=True)]

    ranked_candidates = remove_dup_unigram(ranked_candidates)

    return ranked_candidates


def remove_dup_unigram(ranked_candidates):
    final_ranked = []
    i = 0
    while i < len(ranked_candidates):
        j = 0
        found = False
        while j < i:
            if len(ranked_candidates[i].split()) == 1:
                if ranked_candidates[i] in ranked_candidates[j].split():
                    found = True
                    break
            j += 1
        if not found:
            final_ranked.append(ranked_candidates[i])
        i += 1

    return final_ranked
