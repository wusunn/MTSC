# DMSC-Net Open Source Code
- **results/eval_time**: Inference time measurements (reported in our paper) of the algorithm, exported from MLflow.
- **results/f1_prec_recall**: Accuracy-related metrics (F1, Precision, and Recall) (reported in our paper) , exported from MLflow.
- **checkpoints/**: Checkpoints of our method (seed = 42). The corresponding results can be reproduced and inspected via `reproduce_our_results.sh`.
- **reproduce_our_results.sh**: Script for reproducing the reported accuracy results.

**Note:** In the experimental code, our model is named `M2SFormerATPSelfRegulationSCP2`.


## Datasets
The datasets were obtained from https://www.timeseriesclassification.com/.  After decompression, the directory structure is organized as follows: 
```
dataset/
├── EthanolConcentration
│   ├── EthanolConcentration_TEST.ts
│   └── EthanolConcentration_TRAIN.ts
├── FaceDetection
│   ├── FaceDetection_TEST.ts
│   └── FaceDetection_TRAIN.ts
├── Handwriting
│   ├── Handwriting_TEST.ts
│   └── Handwriting_TRAIN.ts
├── Heartbeat
│   ├── Heartbeat_TEST.ts
│   └── Heartbeat_TRAIN.ts
├── JapaneseVowels
│   ├── JapaneseVowels_TEST.ts
│   └── JapaneseVowels_TRAIN.ts
├── PEMS-SF
│   ├── PEMS-SF_TEST.ts
│   └── PEMS-SF_TRAIN.ts
├── SelfRegulationSCP1
│   ├── SelfRegulationSCP1_TEST.ts
│   └── SelfRegulationSCP1_TRAIN.ts
├── SelfRegulationSCP2
│   ├── SelfRegulationSCP2_TEST.ts
│   └── SelfRegulationSCP2_TRAIN.ts
├── SpokenArabicDigits
│   ├── SpokenArabicDigits_TEST.ts
│   └── SpokenArabicDigits_TRAIN.ts
└── UWaveGestureLibrary
    ├── UWaveGestureLibrary_TEST.ts
    └── UWaveGestureLibrary_TRAIN.ts

```
