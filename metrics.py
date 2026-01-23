import os
from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, cohen_kappa_score, confusion_matrix
import scipy.io as sio
from scipy.io import loadmat, savemat

class AvgrageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.avg = 0
    self.sum = 0
    self.cnt = 0

  def update(self, val, n=1):
    self.sum += val * n
    self.cnt += n
    self.avg = self.sum / self.cnt

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0/batch_size))
    return res, target, pred.squeeze()

def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA

def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=np.float32)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

class NonZeroClipper(object):
    def __call__(self, module):
        if hasattr(module, 'weight'):
           w = module.weight.data
           w.clamp_(1e-6, 1)


def v2(data, out_dir):
        # 2. Extract Prediction (P) and Ground Truth (label)
        prediction = data['P']
        ground_truth = data['label']

        print("GT:")
        plt.figure(figsize=(10,10))
        plt.title("Ground Truth")
        plt.imshow(ground_truth, cmap='jet')   
        plt.axis('off')
        plt.savefig(f"{out_dir}/ground_truth.pdf", 
            format="pdf", 
            dpi=300, 
            bbox_inches="tight")
        plt.show()

        print("PM:")
        plt.figure(figsize=(10,10))
        plt.title("Prediction Map")
        plt.imshow(prediction, cmap='jet')  
        plt.axis('off')
        plt.savefig(f"{out_dir}/prediction_mao.pdf", 
            format="pdf", 
            dpi=300, 
            bbox_inches="tight")
        plt.show()

def print_all_matrics(pre_u, label, total_pos_true, prediction_matrix, dataset, num_classes, class_names, out_dir):
        y_pred = []
        y_true = []

        # Get Perdiction Matrix
        for i in range(total_pos_true.shape[0]):
            y_pred.append(pre_u[i]+1)
            y_true.append(label[total_pos_true[i,0], total_pos_true[i,1]])
            # y_true.append(label[total_pos_true[i,0], total_pos_true[i,1]])

        for i in range(total_pos_true.shape[0]):
          y, x = total_pos_true[i]
          gt_class = int(label[y, x])
          pred_class = int(pre_u[i])

          # Skip unlabeled or invalid ground truth
          if gt_class == 0:
              continue

          # Fill only valid positions
          prediction_matrix[y, x] = pred_class + 1

        savemat(f'{out_dir}/matrix.mat',{'P':prediction_matrix, 'label':label})
        data = loadmat(f'{out_dir}/matrix.mat')

        v2(data, out_dir)

        # Convert to NumPy arrays
        y_pred = np.array(y_pred).astype(int)
        y_true = np.array(y_true).astype(int)

        # Remove unlabeled class
        mask = np.array(y_true) != 0 
        y_true = np.array(y_true)[mask]
        y_pred = np.array(y_pred)[mask]

        print("Unique classes in y_true:", np.unique(y_true))
        print("Unique classes in y_pred:", np.unique(y_pred))
        print("Length of class_names:", len(class_names))


        print("\n Classification Report:\n")
        print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

        cm = confusion_matrix(y_true, y_pred)

        # Per-class accuracy: diagonal / row sum
        per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
        support = cm.sum(axis=1)

        # Overall Accuracy
        OA = accuracy_score(y_true, y_pred)

        # Average Accuracy
        AA = np.mean(per_class_accuracy)

        # Kappa
        kappa = cohen_kappa_score(y_true, y_pred)

        # Print nicely
        print("\nPer-Class Accuracy:")
        print(f"{'Class':<28}{'Support':>12}{'Accuracy':>12}")
        print("-" * 53)

        for i, acc in enumerate(per_class_accuracy):
            print(f"{class_names[i]:<28}{support[i]:>12d}{acc*100:>12.2f}")

        print("\nEvaluation Metrics on Test Set:")
        print(f"{'Overall Accuracy (OA):':<25}{OA*100:>8.2f}%")
        print(f"{'Average Accuracy (AA):':<25}{AA*100:>8.2f}%")
        print(f"{'Kappa Coefficient:':<25}{kappa*100:>8.2f}%")

        # Confusion matrix (already computed as `cm`)
        print("\nConfussion Matrix: ")
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Confusion Matrix with Class Names")
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/confusion_matrix.pdf", 
                    format="pdf", 
                    dpi=300, 
                    bbox_inches="tight")
        plt.show()