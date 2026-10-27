T229 - xBD Disaster Building Damage Assessment

PROJECT OVERVIEW
This project explores post-disaster building damage assessment using
pre- and post-disaster imagery from the xBD dataset.

COMPONENTS
1. ResNet18 building-level damage classifier.
2. Experimental Siamese U-Net segmentation branch.
3. Example damage heatmaps, overlays, per-building predictions,
   validation predictions, and evaluation reports.

RESNET18 CLASSIFIER
Validation results:
- Samples: 2,886
- Accuracy: 0.8829
- Macro-F1: 0.8836

These are validation results, not independent test-set results.
Performance may vary on unseen disaster scenes.

SIAMESE U-NET SEGMENTATION
The segmentation branch is experimental. Its validation results show
weak performance for minor- and major-damage classes. Different
evaluation runs and methods produced inconsistent metrics, so those
scores should not be treated as a single verified final result.
The segmentation outputs and heatmaps are illustrative predictions,
not proof of reliable end-to-end damage mapping.

OUTPUTS
The package includes evaluation CSV files, confusion matrix,
validation predictions, example overlays and heatmaps, per-building
prediction files, a manifest, and this README.

LIMITATIONS
- Validation metrics do not guarantee performance on unseen disasters.
- Building-level classification and pixel-level segmentation metrics
  measure different tasks and should not be compared directly.
- Geographic coordinates and map alignment have not been fully verified.
- Predictions should be reviewed before operational or emergency use.

This package represents a prototype for disaster damage assessment.
