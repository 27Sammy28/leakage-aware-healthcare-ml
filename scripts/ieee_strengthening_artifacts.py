import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
results = pd.read_csv('benchmark_results.csv')
def metrics_from_counts(tn, fp, fn, tp):
    n = tn + fp + fn + tp
    acc = (tn + tp) / n
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-15)
    spec = tn / max(tn + fp, 1)
    bal = 0.5 * (rec + spec)
    denom = np.sqrt(max((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn), 1))
    mcc = (tp * tn - fp * fn) / denom
    return acc, f1, rec, spec, bal, mcc

rows=[]
for _, r in results[results['sampling'].eq('Default')].iterrows():
    counts=np.array([r.tn, r.fp, r.fn, r.tp], dtype=int)
    probs=counts/counts.sum()
    boots=[]
    for _ in range(2000):
        tn, fp, fn, tp = rng.multinomial(int(counts.sum()), probs)
        boots.append(metrics_from_counts(tn, fp, fn, tp))
    boots=np.array(boots)
    point=metrics_from_counts(*counts)
    for idx, metric in enumerate(['accuracy','f1','recall','specificity','balanced_accuracy','mcc']):
        lo, hi = np.quantile(boots[:,idx], [0.025,0.975])
        rows.append({'model':r.model,'metric':metric,'point':point[idx],'ci_low':lo,'ci_high':hi})
ci=pd.DataFrame(rows)
ci.to_csv('bootstrap_ci_threshold_metrics.csv', index=False)

# compact manuscript table for leading metrics
compact=[]
for model in ['Decision Tree','LR','Linear SVM','Nearest Centroid','Gaussian NB']:
    sub=ci[(ci.model==model) & (ci.metric.isin(['accuracy','f1','balanced_accuracy','mcc']))]
    vals={m: sub[sub.metric==m].iloc[0] for m in sub.metric.unique()}
    compact.append({
        'Model': model,
        'Accuracy 95% CI': f"{vals['accuracy'].point:.3f} ({vals['accuracy'].ci_low:.3f}--{vals['accuracy'].ci_high:.3f})",
        'F1 95% CI': f"{vals['f1'].point:.3f} ({vals['f1'].ci_low:.3f}--{vals['f1'].ci_high:.3f})",
        'Balanced acc. 95% CI': f"{vals['balanced_accuracy'].point:.3f} ({vals['balanced_accuracy'].ci_low:.3f}--{vals['balanced_accuracy'].ci_high:.3f})",
        'MCC 95% CI': f"{vals['mcc'].point:.3f} ({vals['mcc'].ci_low:.3f}--{vals['mcc'].ci_high:.3f})",
    })
pd.DataFrame(compact).to_csv('bootstrap_ci_compact.csv', index=False)

# fixed-threshold decision-curve-style net benefit at pt=0.50
nb=[]
pt=0.50
for _, r in results[results['sampling'].eq('Default')].iterrows():
    n = r.tn + r.fp + r.fn + r.tp
    net_benefit = r.tp/n - r.fp/n * (pt/(1-pt))
    nb.append({'model':r.model,'threshold':pt,'net_benefit':net_benefit})
pd.DataFrame(nb).to_csv('fixed_threshold_net_benefit.csv', index=False)
print('wrote bootstrap_ci_threshold_metrics.csv, bootstrap_ci_compact.csv, fixed_threshold_net_benefit.csv')
print(pd.DataFrame(compact).to_string(index=False))
print(pd.DataFrame(nb).sort_values('net_benefit', ascending=False).to_string(index=False))
