from tqdm.auto import tqdm

from utils import compute_rel, compute_prf, compute_averagep


def evaluate_model(candidates, references, n_best=10, return_dict=False):
    precisions = []
    recalls = []
    f_scores = []
    average_p = []

    for doc_id in tqdm(candidates):
        if doc_id not in references:
            continue

        processed_candidates = [[" ".join(x[0].split("-"))] for x in candidates[doc_id]]
        processed_references = [[" ".join(x[0].split("-"))] for x in references[doc_id]]

        rel = compute_rel(processed_candidates, processed_references)

        if n_best == -1:
            max_len = len(processed_references)
        else:
            max_len = min(len(processed_candidates), n_best)

        p, r, f = compute_prf(rel[:max_len], processed_references)
        ap = compute_averagep(rel, processed_references)

        precisions.append(p)
        recalls.append(r)
        f_scores.append(f)
        average_p.append(ap)

    P = sum(precisions) / len(precisions) * 100.0
    R = sum(recalls) / len(recalls) * 100.0
    F = sum(f_scores) / len(f_scores) * 100.0
    MAP = sum(average_p) / len(average_p) * 100.0

    print("| {1:5.2f} | {2:5.2f} | {3:5.2f} | {4:5.2f} | {0:2d} |".format(n_best, P, R, F, MAP))
    if return_dict:
        return P, R, F, MAP, n_best
