import json
import numpy as np
from scipy import stats

with open("/Users/sukhmansinghnarula/.claude-science/orgs/d2396652-2759-4667-bd71-2839c2fe85f6/artifacts/proj_9d30ad28589c/89146e04-c94a-4b58-8c6a-e5184442cecc/vc4eb1d63_results.json") as f:
    saved = json.load(f)
results = saved["results"]

binom_results = {}
for k, v in results.items():
    n_test = sum(sum(row) for row in v["confusion_matrix"])
    n_correct = round(v["test_acc"] * n_test)
    pval = stats.binomtest(n_correct, n_test, 1/3, alternative='two-sided').pvalue
    ci = stats.binomtest(n_correct, n_test, 1/3).proportion_ci(confidence_level=0.95)
    binom_results[k] = {"n_test": n_test, "n_correct": n_correct, "p_value_vs_1_3": pval,
                         "ci_95": [ci.low, ci.high]}

with open("binom_results.json", "w") as f:
    json.dump(binom_results, f, indent=2)