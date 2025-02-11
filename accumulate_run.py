import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def compute_fraction_above_threshold(npy_file, threshold):
    """Loads a .npy file, computes fraction of entries > threshold."""
    data = np.load(npy_file)
    fraction = np.mean(data > threshold)
    return fraction

def compute_eps_fraction_element(npy_file, eps):
    """Loads a .npy file and returns the eps-fraction-th element."""
    data = np.load(npy_file)
    sorted_data = np.sort(data)
    index = int((1-eps) * len(data))
    return sorted_data[index]

def plot_roc_curve(fpr, tpr, auc_score, save_path="roc_curve.png"):
    """Plots and saves the ROC curve."""
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    
    # Save the figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"ROC curve saved to {save_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Compute fraction of array entries greater than threshold.")
    parser.add_argument("npy_files", nargs="+", help="List of .npy files to process")
    parser.add_argument("threshold", type=float, help="Threshold value")
    parser.add_argument("eps", type=float, help="Farness parameter")

    args = parser.parse_args()
    
    positive_elems = []
    negative_elems = []
    for npy_file in args.npy_files:
        if os.path.exists(npy_file):
            # fraction = compute_fraction_above_threshold(npy_file, args.threshold)
            # print(f"{npy_file}: {fraction:.6f} : {fraction >= args.eps}")
            epsmedian = compute_eps_fraction_element(npy_file, args.eps)
            print(f"{npy_file}: {epsmedian:.6f} : {epsmedian <= args.threshold}")
            if npy_file.startswith("deepseek"):
                positive_elems.append(epsmedian)
            elif npy_file.startswith("stability"):
                negative_elems.append(epsmedian)
        else:
            print(f"Error: {npy_file} not found.")

    # Assuming positive and negative elements are the true labels and scores
    y_true = [1] * len(positive_elems) + [0] * len(negative_elems)  # 1 for positive, 0 for negative
    y_scores = positive_elems + negative_elems  # Scores from both categories
    
    # Compute ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    print(roc_auc)
    
    # Plot ROC curve
    plot_roc_curve(fpr, tpr, roc_auc, save_path=f"roc_curve_{args.eps}.png")

if __name__ == "__main__":
    main()
