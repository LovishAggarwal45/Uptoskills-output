"""
Quantitative Evaluation Engine for VideoMAE Crowd Outlier Localization.

Calculates exact benchmark performance metrics across all ground-truth video scenarios:
- ROC-AUC & PR-AUC
- Frame-level Classification Accuracy, Precision, Recall, F1-Score
- False Positive Rate (FPR) / False Alarm Rate
- Optimal Threshold Calibration (Youden's J-Index / Max F1-Score)
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_curve
from detector.outlier_detector import CrowdOutlierDetector


def get_ground_truth_labels():
    """Returns frame-level ground-truth binary labels (0 = Normal, 1 = Anomaly) for each benchmark video."""
    return {
        "data/scenario_vehicle_intrusion.mp4": {
            "title": "High-Speed Vehicle Intrusion",
            "anomaly_ranges": [(60, 130)],
            "total_frames": 200,
            "category": "High-Speed Motion Anomaly"
        },
        "data/scenario_stampede.mp4": {
            "title": "Stampede & Bottleneck Blockade",
            "anomaly_ranges": [(60, 130)],
            "total_frames": 200,
            "category": "High-Speed Motion Anomaly"
        },
        "data/scenario_night_surge.mp4": {
            "title": "Night Vision Crowd Surge",
            "anomaly_ranges": [(55, 135)],
            "total_frames": 200,
            "category": "High-Speed Motion Anomaly"
        },
        "data/sample_crowd_test.mp4": {
            "title": "Baseline Panic & Counter-Flow",
            "anomaly_ranges": [(70, 120), (170, 210)],
            "total_frames": 240,
            "category": "Spatio-Temporal Flow Anomaly"
        },
        "data/scenario_multizone.mp4": {
            "title": "Multi-Zone Concurrent Disturbances",
            "anomaly_ranges": [(45, 95), (125, 175)],
            "total_frames": 220,
            "category": "Multi-Zone Localized Anomaly"
        },
        "data/scenario_fire_evacuation.mp4": {
            "title": "Fire Alarm Panic Evacuation",
            "anomaly_ranges": [(50, 140)],
            "total_frames": 200,
            "category": "Crowd Directional Shift"
        },
        "data/scenario_crowd_freeze.mp4": {
            "title": "Sudden Crowd Freeze Anomaly",
            "anomaly_ranges": [(60, 130)],
            "total_frames": 200,
            "category": "Static Crowd Halt"
        }
    }


def evaluate_system():
    detector = CrowdOutlierDetector(config_path="config.yaml")
    gt_info = get_ground_truth_labels()

    all_y_true = []
    all_y_scores = []
    scenario_metrics = []

    print("=" * 80)
    print("     EVALUATING VIDEOMAE CROWD OUTLIER LOCALIZATION PERFORMANCE     ")
    print("=" * 80)

    for vpath, meta in gt_info.items():
        if not os.path.exists(vpath):
            continue

        title = meta["title"]
        total_frames = meta["total_frames"]
        y_true = np.zeros(total_frames, dtype=int)
        for start_f, end_f in meta["anomaly_ranges"]:
            y_true[start_f : end_f + 1] = 1

        # Run detector
        out_vpath = f"outputs/eval_{os.path.basename(vpath)}"
        _, scores = detector.process_video(vpath, output_path=out_vpath)

        n_frames = min(len(y_true), len(scores))
        y_true = y_true[:n_frames]
        scores = scores[:n_frames]

        roc_auc = roc_auc_score(y_true, scores)
        precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, scores)
        pr_auc = auc(recall_arr, precision_arr)

        # Compute optimal decision threshold via Max F1-Score
        f1_scores = [2 * (p * r) / (p + r + 1e-6) for p, r in zip(precision_arr, recall_arr)]
        best_idx = np.argmax(f1_scores)
        best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else scores.mean()

        y_pred = (scores >= best_thresh).astype(int)

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        norm_mse = scores[y_true == 0].mean() if (y_true == 0).any() else 1.0
        anom_mse = scores[y_true == 1].mean() if (y_true == 1).any() else 1.0
        snr = anom_mse / max(norm_mse, 1e-6)

        scenario_metrics.append({
            "Scenario": title,
            "Category": meta["category"],
            "Frames": n_frames,
            "ROC-AUC": roc_auc,
            "PR-AUC": pr_auc,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1-Score": f1,
            "Optimal Thresh": best_thresh
        })

        all_y_true.extend(y_true)
        all_y_scores.extend(scores)

    # Calculate overall dataset performance metrics
    all_y_true = np.array(all_y_true)
    all_y_scores = np.array(all_y_scores)

    overall_roc_auc = roc_auc_score(all_y_true, all_y_scores)
    precision_arr, recall_arr, thresholds = precision_recall_curve(all_y_true, all_y_scores)
    overall_pr_auc = auc(recall_arr, precision_arr)

    f1_scores = [2 * (p * r) / (p + r + 1e-6) for p, r in zip(precision_arr, recall_arr)]
    best_idx = np.argmax(f1_scores)
    overall_best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else all_y_scores.mean()

    all_y_pred = (all_y_scores >= overall_best_thresh).astype(int)

    overall_acc = accuracy_score(all_y_true, all_y_pred)
    overall_prec = precision_score(all_y_true, all_y_pred, zero_division=0)
    overall_rec = recall_score(all_y_true, all_y_pred, zero_division=0)
    overall_f1 = f1_score(all_y_true, all_y_pred, zero_division=0)

    tn, fp, fn, tp = confusion_matrix(all_y_true, all_y_pred).ravel()
    fpr = fp / (fp + tn + 1e-6)

    # High-Motion Anomaly Sub-Group Performance (Vehicle Intrusion, Stampede, Night Surge)
    motion_scenarios = ["High-Speed Vehicle Intrusion", "Stampede & Bottleneck Blockade", "Night Vision Crowd Surge"]
    df_metrics = pd.DataFrame(scenario_metrics)
    motion_df = df_metrics[df_metrics["Scenario"].isin(motion_scenarios)]

    motion_roc_auc = motion_df["ROC-AUC"].mean()
    motion_pr_auc = motion_df["PR-AUC"].mean()
    motion_acc = motion_df["Accuracy"].mean()
    motion_f1 = motion_df["F1-Score"].mean()

    print("\n[PER-SCENARIO ACCURACY BREAKDOWN]")
    print(df_metrics[["Scenario", "ROC-AUC", "PR-AUC", "Accuracy", "Precision", "Recall", "F1-Score"]].to_string(index=False))

    print("\n" + "=" * 80)
    print("                      OVERALL SYSTEM ACCURACY & PERFORMANCE                ")
    print("=" * 80)
    print(f"  • High-Speed Motion Anomaly ROC-AUC:    {motion_roc_auc * 100:.2f}%  (Vehicle Intrusion, Stampede, Surge)")
    print(f"  • High-Speed Motion Anomaly PR-AUC:     {motion_pr_auc * 100:.2f}%")
    print(f"  • High-Speed Motion Classification Acc: {motion_acc * 100:.2f}%")
    print(f"  • High-Speed Motion Anomaly F1-Score:   {motion_f1 * 100:.2f}%")
    print("  " + "-" * 76)
    print(f"  • Overall System ROC-AUC Score:         {overall_roc_auc * 100:.2f}%  (Across All 7 Benchmark Scenarios)")
    print(f"  • Overall Precision-Recall AUC (PR-AUC):{overall_pr_auc * 100:.2f}%")
    print(f"  • Overall Frame Classification Accuracy:{overall_acc * 100:.2f}%")
    print(f"  • Overall Precision:                    {overall_prec * 100:.2f}%")
    print(f"  • Overall Recall / Sensitivity:         {overall_rec * 100:.2f}%")
    print(f"  • Overall F1-Score:                     {overall_f1 * 100:.2f}%")
    print(f"  • False Positive Rate (FPR):            {fpr * 100:.2f}%")
    print(f"  • Total Evaluated Video Frames:         {len(all_y_true)}")
    print("=" * 80)

    # Save summary dataframe
    df_metrics.to_csv("outputs/evaluation_metrics.csv", index=False)
    print(f"\n[Saved] Quantitative evaluation metrics exported to: outputs/evaluation_metrics.csv")


if __name__ == "__main__":
    evaluate_system()
