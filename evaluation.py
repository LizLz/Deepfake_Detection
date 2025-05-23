import numpy as np
from sklearn.metrics import roc_curve, f1_score
from scipy.special import expit  # sigmoid function

all_scores = []
all_labels = []

model.eval()
with torch.no_grad():
    for features, labels in dev_dataLoader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        
        all_scores.extend(outputs.cpu().numpy().flatten())  # model scores (probabilities)
        all_labels.extend(labels.cpu().numpy())             # ground truth labels
non_zero_count = np.count_nonzero(all_scores)
print(f"Number of non-zero scores: {non_zero_count}")

def calculate_eer(labels, scores):
    '''
    EER is the point where: False Acceptance Rate (FAR) ≈ False Rejection Rate (FRR)
    The percentage of fake samples accepted as real ≈ real samples rejected as fake
    Lower EER = better model.
    '''

    # fpr corresponds to FAR
    # threshold is the cutoff value (e.g., from EER) used to decide whether a score is positive (1) or negative (0).
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1) 

    # fnr corresponds to FRR
    fnr = 1 - tpr 

    #Calculates the index where the absolute difference between FNR and FPR is smallest.
    eer_threshold_idx = np.nanargmin(np.abs(fnr - fpr)) 

    eer = (fpr[eer_threshold_idx] + fnr[eer_threshold_idx]) / 2
    
    return eer * 100, thresholds[eer_threshold_idx]

eer, threshold = calculate_eer(np.array(all_labels), np.array(all_scores))

# transform logit threshold to prob threshold
prob_threshold = expit(threshold)

print(f"EER: {eer:.2f}%, Logit threshold at EER: {threshold:.4f}, Prob threshold at EER: {prob_threshold:.4f}")


# Binarize predictions using threshold，True for predicted class 1 (authentic), and False for class 0 (fake)
preds = (all_scores >= threshold).astype(int)

# Calculate F1 Score
f1 = f1_score(all_labels, preds)
print(f"F1 Score: {f1:.4f}")