from tqdm.auto import tqdm

from utils import compute_rel, compute_prf, compute_averagep


def evaluate_model(candidates, references, n_best=10, present=False, absent=False):
    precisions = []
    recalls = []
    f_scores = []
    average_p = []

    for doc_id in tqdm(candidates):
        if doc_id not in references:
            continue
        rel = compute_rel(candidates[doc_id], references[doc_id])

        max_len = min(len(candidates[doc_id]), n_best)

        p, r, f = compute_prf(rel[:max_len], references[doc_id])
        ap = compute_averagep(rel, references[doc_id])

        precisions.append(p)
        recalls.append(r)
        f_scores.append(f)
        average_p.append(ap)

    P = sum(precisions) / len(precisions) * 100.0
    R = sum(recalls) / len(recalls) * 100.0
    F = sum(f_scores) / len(f_scores) * 100.0
    MAP = sum(average_p) / len(average_p) * 100.0

    print("| {1:5.2f} | {2:5.2f} | {3:5.2f} | {4:5.2f} | {0:2d} |".format(n_best, P, R, F, MAP))
