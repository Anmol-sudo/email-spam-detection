# Phase 9 and 10 Code

_Since you're wrapping up the notebook manually, just create a few new cells at the end of your Jupyter / Colab notebook and copy-paste the code blocks below._

---

## Cell 1: Markdown (Phase 9 Header)

```markdown
---

## Phase 9 — Feature Importance & LIME Explainability

> LIME (Local Interpretable Model-agnostic Explanations) explains **why** the model made a specific prediction by perturbing the input and observing how predictions change.
```

## Cell 2: Code (Install & Setup LIME)

```python
# Install LIME (uncomment if not installed)
# !pip install lime -q
import lime
import lime.lime_text
from lime.lime_text import LimeTextExplainer
print('LIME imported successfully!')

# Set up LIME explainer
class_names = ['Spam', 'Ham']
explainer = LimeTextExplainer(class_names=class_names)

def lime_predict(texts):
    """Prediction function for LIME — takes raw text list, returns probabilities."""
    cleaned = [clean_text(t) for t in texts]
    return best_tuned.predict_proba(cleaned)

print('LIME explainer ready!')
```

## Cell 3: Code (LIME Explanation for SPAM)

```python
# ── LIME Explanation for a SPAM message ───────────────────────────
spam_msg = "FREE entry in 2 a wkly comp to win FA Cup final tkts! Text FA to 87121. £250 prize guaranteed!"

print(f'Explaining: "{spam_msg}"\n')
exp_spam = explainer.explain_instance(
    spam_msg,
    lime_predict,
    num_features=12,
    num_samples=500
)

print('Top features and their contribution to SPAM vs HAM:')
for feat, weight in exp_spam.as_list():
    direction = '→ SPAM' if weight < 0 else '→ HAM'
    bar = '█' * int(abs(weight) * 50)
    print(f'  {feat:<20} {weight:+.4f}  {direction}  {bar}')

# Show in notebook as HTML widget
exp_spam.show_in_notebook(text=True)
```

## Cell 4: Code (LIME Explanation for HAM)

```python
# ── LIME Explanation for a HAM message ────────────────────────────
ham_msg = "Hey, are you coming to the party tonight? Let me know when you're free!"

print(f'Explaining: "{ham_msg}"\n')
exp_ham = explainer.explain_instance(
    ham_msg,
    lime_predict,
    num_features=12,
    num_samples=500
)

print('Top features and their contribution:')
for feat, weight in exp_ham.as_list():
    direction = '→ SPAM' if weight < 0 else '→ HAM'
    bar = '█' * int(abs(weight) * 50)
    print(f'  {feat:<20} {weight:+.4f}  {direction}  {bar}')

exp_ham.show_in_notebook(text=True)
```

## Cell 5: Code (LIME Bar Charts)

```python
# ── Side-by-side LIME bar charts ──────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, exp, title, msg in zip(
    axes,
    [exp_spam, exp_ham],
    ['🚨 SPAM Message LIME Explanation', '✅ HAM Message LIME Explanation'],
    [spam_msg, ham_msg]
):
    feats  = [f for f, w in exp.as_list()]
    weights = [w for f, w in exp.as_list()]
    colors  = ['#e74c3c' if w < 0 else '#2ecc71' for w in weights]
    bars = ax.barh(feats[::-1], weights[::-1], color=colors[::-1], edgecolor='black', alpha=0.85)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('LIME Weight (negative=spam, positive=ham)')
    ax.set_xlim(-max(abs(w) for w in weights)*1.3, max(abs(w) for w in weights)*1.3)

plt.suptitle('LIME Local Explanations — Why did the model classify this?',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
print('\nRed bars = words pushing toward SPAM | Green bars = words pushing toward HAM')
```

## Cell 6: Code (Global Feature Importance)

```python
# ── Global Feature Importance (LR Coefficients) ───────────────────
lr_clf    = best_tuned.named_steps['clf']
tfidf_vec = best_tuned.named_steps['tfidf']
feat_names = np.array(tfidf_vec.get_feature_names_out())
coef       = lr_clf.coef_[0]

N = 20
spam_idx = coef.argsort()[:N]
ham_idx  = coef.argsort()[-N:][::-1]

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].barh(feat_names[spam_idx][::-1], np.abs(coef[spam_idx][::-1]),
             color='#e74c3c', edgecolor='black', alpha=0.85)
axes[0].set_title('🚨 Top 20 SPAM-Indicating Features\n(LR Coefficients)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('|Coefficient Value|')

axes[1].barh(feat_names[ham_idx][::-1], coef[ham_idx][::-1],
             color='#2ecc71', edgecolor='black', alpha=0.85)
axes[1].set_title('✅ Top 20 HAM-Indicating Features\n(LR Coefficients)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Coefficient Value')

plt.suptitle('Global Feature Importance — Logistic Regression', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()
```

## Cell 7: Markdown (Phase 10 Header)

```markdown
---

## Phase 10 — Final Report & Project Summary

> Complete summary of all phases, final model performance, and conclusions.
```

## Cell 8: Code (Final Metrics)

```python
# ── Final Model Evaluation: Tuned LR on full test set ─────────────
final_preds = best_tuned.predict(X_test_c)
final_proba = best_tuned.predict_proba(X_test_c)[:, 1]

final_metrics = {
    'Accuracy'  : accuracy_score(Y_test_c, final_preds),
    'Precision' : precision_score(Y_test_c, final_preds),
    'Recall'    : recall_score(Y_test_c, final_preds),
    'F1-Score'  : f1_score(Y_test_c, final_preds),
    'ROC-AUC'   : roc_auc_score(Y_test_c, final_proba),
}

print('=' * 55)
print('  FINAL MODEL PERFORMANCE REPORT')
print('  Model: Tuned Logistic Regression (GridSearchCV)')
print('=' * 55)
for metric, val in final_metrics.items():
    bar = '█' * int(val * 40)
    print(f'  {metric:<12}: {val*100:6.2f}%  {bar}')
print('=' * 55)
print(classification_report(Y_test_c, final_preds, target_names=['Spam','Ham']))
```

## Cell 9: Code (Baseline vs Final Comparison)

```python
# ── Final comparison: Baseline vs Best Tuned ──────────────────────
comparison_data = {
    'Model'          : ['Baseline LR (raw TF-IDF)', 'Tuned LR (NLTK + bigrams + GridSearch)'],
    'Accuracy (%)'   : [baseline_test_acc*100, final_metrics['Accuracy']*100],
    'Precision (%)'  : [precision_score(Y_test_c, lr_base.predict(X_test_base))*100, final_metrics['Precision']*100],
    'Recall (%)'     : [recall_score(Y_test_c, lr_base.predict(X_test_base))*100, final_metrics['Recall']*100],
    'F1-Score (%)'   : [f1_score(Y_test_c, lr_base.predict(X_test_base))*100, final_metrics['F1-Score']*100],
    'ROC-AUC (%)'    : [roc_auc_score(Y_test_c, lr_base.predict_proba(X_test_base)[:,1])*100, final_metrics['ROC-AUC']*100],
}
cdf = pd.DataFrame(comparison_data).set_index('Model')
print('=== Baseline vs Final Model ===')
print(cdf.round(2).to_string())

# Improvement summary
print('\n=== Improvement ===')
for col in cdf.columns:
    delta = cdf.iloc[1][col] - cdf.iloc[0][col]
    print(f'  {col:<16}: {delta:+.2f}%')
```

## Cell 10: Code (Radar Chart)

```python
# ── Radar Chart — Baseline vs Tuned Model ─────────────────────────
from matplotlib.patches import FancyArrowPatch

categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
N_cat = len(categories)
angles = np.linspace(0, 2*np.pi, N_cat, endpoint=False).tolist()
angles += angles[:1]

baseline_vals = [baseline_test_acc,
                 precision_score(Y_test_c, lr_base.predict(X_test_base)),
                 recall_score(Y_test_c, lr_base.predict(X_test_base)),
                 f1_score(Y_test_c, lr_base.predict(X_test_base)),
                 roc_auc_score(Y_test_c, lr_base.predict_proba(X_test_base)[:,1])]
tuned_vals    = list(final_metrics.values())

baseline_vals += baseline_vals[:1]
tuned_vals    += tuned_vals[:1]

fig, ax = plt.subplots(figsize=(8,8), subplot_kw=dict(polar=True))
ax.plot(angles, baseline_vals, 'o--', color='#95a5a6', linewidth=2, label='Baseline LR')
ax.fill(angles, baseline_vals, alpha=0.15, color='#95a5a6')
ax.plot(angles, tuned_vals, 'o-', color='#3498db', linewidth=2, label='Tuned LR')
ax.fill(angles, tuned_vals, alpha=0.25, color='#3498db')
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, fontsize=12)
ax.set_ylim(0.85, 1.0)
ax.set_title('Baseline vs Tuned Model\nPerformance Radar', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), fontsize=11)
plt.tight_layout()
plt.show()
```

## Cell 11: Markdown (Project Conclusion)

```markdown
---
## 🏁 Project Conclusions

### What we built:
A **production-ready spam detection pipeline** that goes far beyond a basic classifier.

### Key Takeaways:

| # | Insight |
|---|---|
| 1 | **Accuracy alone is misleading** on imbalanced data — always use F1, Recall, and ROC-AUC |
| 2 | **NLTK preprocessing** + bigrams gave meaningful improvement over raw TF-IDF |
| 3 | **Linear SVM and LR** are the best models for this task — fast, accurate, interpretable |
| 4 | **SMOTE** improved spam recall — critical for a spam filter that must catch all spam |
| 5 | **LIME** makes the black-box ML explainable — we can see exactly which words triggered spam |
| 6 | The model is now **deployable** via Streamlit with a `.pkl` pipeline |

### Future Work:
- Fine-tune a **BERT / DistilBERT** model for even higher accuracy
- Test generalization on the **Enron email dataset** (30k+ real emails)
- Add **email header analysis** (sender domain, reply-to mismatch) as features
- Deploy to **cloud** (Streamlit Cloud / Hugging Face Spaces)

---

_Project by Arjun | Internship 2026 | All phases complete ✅_
```
