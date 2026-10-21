# Plain-language explainer

Task: from 13 clinical measurements of a patient, predict whether heart disease is present (1) or absent (0). 303 patients. Numbers quoted come from `results/` (70/30 split or 10-fold CV, see README).

## Why preprocessing is done inside a Pipeline
Imputing missing values and standardising use statistics (median, mean, sd). If you compute them on all 303 patients, test patients have influenced the training pipeline: *leakage*, which makes scores look better than they are. A scikit-learn `Pipeline` re-learns those statistics on the training part each time, then only applies them to the test part. (The script shows scaled train columns have mean 0, test columns do not.)

## The models

**Decision Tree.** Asks a chain of yes/no questions ("thal <= 4.5?", "cp <= 3.5?"), each chosen to split patients into purer groups, until leaves are pure. Key hyper-parameter: `criterion` (entropy/Gini) and depth. Unrestricted depth means it memorises the training set (train accuracy 1.000) and generalises worst here (≈0.73). It loses because 212 training patients give noisy splits deep in the tree; pruning (`max_depth`) or a forest would help. Good for explaining *why* a prediction was made.

**Naive Bayes (Gaussian).** For each class it learns a bell curve per feature, then multiplies the likelihoods of a patient's values by the class prior ("naively" assuming features are independent) and picks the likelier class. No important hyper-parameter. Fast and stable on small data, which is why it does well here (≈0.84-0.88) although the independence assumption is false (e.g. `thal`, `cp`, `exang` are related).

**Random Forest.** Many trees, each trained on a random resample of patients and a random subset of features at every split; vote by majority. Averaging cancels much of a single tree's noise. Key hyper-parameters: number of trees, `max_depth`, `min_samples_leaf`, `max_features`. Still fits training data perfectly by default (train 1.0 vs test ≈0.82-0.87); small tuning helped little (+0.004 in CV), within noise. Feature importances: `cp`, `thalach`, `ca`, `thal`, `oldpeak` lead.

**KNN.** To classify a patient, find the *k* most similar training patients and take the majority vote. No training in the usual sense. Key hyper-parameters: *k* and the distance. Small k = jumpy (k=1: ≈0.73), large k = over-smoothed. Needs standardised features, otherwise `chol` dominates distance. It is competitive here (≈0.83-0.87) but not better than logistic regression or Naive Bayes; with 13 mixed features and one-hot dummies, "similar" is not very well defined.

**SVM.** Finds the dividing surface with the widest margin between the two classes. Key hyper-parameters: kernel and `C` (how much to penalise points on the wrong side; I kept C=1). Works well on small standardised data (≈0.82-0.86), and linear vs rbf differ by little.

**Logistic Regression.** Computes a weighted sum of the features and turns it into a probability with an S-shaped curve. Key hyper-parameter: regularisation strength `C` (default). Its coefficients are interpretable (e.g. male sex, asymptomatic chest pain, number of vessels `ca` raise the odds of disease). Because the signal in this dataset is mostly roughly linear, it ties for best and often wins; simple models are hard to beat on 303 patients.

## Information gain and Gini
A node with 50/50 patients is maximally impure; a node with all one class is pure.
* **Entropy** $H=-\sum p_c\log_2 p_c$ (bits of uncertainty). **Information gain** = entropy of the parent − size-weighted average entropy of the children.
* **Gini** $G=1-\sum p_c^2$ (chance two random patients from the node have different classes). The split with the largest decrease is chosen.

Worked example from the script (70/30 training part, 212 patients: 115 absent, 97 present): parent entropy 0.9948 bits. Question `thal <= 4.5` makes children of 121 patients (entropy 0.7659) and 91 patients (entropy 0.7793). Information gain = 0.9948 − (121/212 · 0.7659 + 91/212 · 0.7793) = **0.2231 bits**, the largest of any possible root question, so it becomes the root. Entropy and Gini nearly always rank splits the same way; here entropy trees scored slightly higher in all protocols (CV 0.728 vs 0.692), but with std 0.078 this is not a reliable difference.

## Distance in KNN
* **Euclidean:** straight-line distance $\sqrt{\sum (a_i-b_i)^2}$; a single big difference dominates.
* **Manhattan:** $\sum |a_i-b_i|$, walking along axes; each feature counts in proportion to its difference.
With one-hot dummies Manhattan is often a bit more robust. At k=8 10-fold CV: Euclidean 0.838, Manhattan 0.815, but their best-k CV accuracies are 0.845 and 0.851: no clear winner.

## SVM kernels
A kernel measures similarity between two patients, letting a linear method draw a curved boundary.
* `linear`: a flat plane. `poly`: polynomial curves (degree 3). `rbf`: similarity decays with distance, giving smooth islands. `sigmoid`: tanh shape, rarely best.
The more flexible kernels can fit finer boundaries but overfit more easily with few patients. Here all four are within 0.03 of each other in CV.

## Metrics (positive = disease present)
Confusion matrix cells: TP (disease, predicted disease), FN (disease missed), FP (healthy flagged), TN.
* **Accuracy** = (TP+TN)/all. Fine when classes are balanced (here 139 vs 164) but hides which kind of error happens.
* **Precision** = TP/(TP+FP): of patients flagged as diseased, how many really are.
* **Recall / sensitivity** = TP/(TP+FN): of the diseased patients, how many we catch. In medicine missing disease (FN) is usually worse than a false alarm, so recall matters.
* **F1** = 2PR/(P+R), harmonic mean of precision and recall; always between them.
* **ROC-AUC:** plots recall vs false-positive rate as the decision threshold moves; the area is the probability that a random diseased patient gets a higher score than a random healthy one (0.5 = guessing, 1 = perfect). Threshold-free, so it can differ from accuracy.
* **Train vs test:** a model with train accuracy ≫ test accuracy (DT, RF) is overfitting.
* **Sampling noise:** with 91 test patients one patient = 1.1 points and the 95% interval on an accuracy of ~0.85 is about ±7 points, so small gaps between models are not evidence of a real difference.
