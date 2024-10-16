Overview:
This Jupyter notebook is part of a project involving speech recognition for air communication. It demonstrates the process of extracting features from audio data using the wav2vec2 model and analyzing the feature contributions using LIME and SHAP explainability tools.
The purpose of this notebook is to identify which features (or dimensions) extracted from the audio data are most important for prediction, so that the final deep learning model can focus on those critical features.

Key Steps in the Notebook
1. Loading Pre-Extracted Features
The features used in this notebook were extracted from audio data using the wav2vec2 model. The extracted features are high-dimensional vectors representing abstract patterns in the audio, not simple features like "pitch" or "volume." Therefore, the features do not have explicit names or labels.
the dataset is loaded and the train and test files present in the data were in the binary format which was converted to audio format as .wav files of both train files and test file as originally there were 4 train files and 1 test file and the train files .wav audio files were saved to output_audio_files
and test .wav audio file is saved to the test_audio_file

2. Using wav2vec2 2.0 model for feature extraction
 wav2vec2 2.0 model is used to ectract the features of the audio file and The features extracted by wav2vec2 are complex, high-dimensional, and do not correspond to intuitive feature names (like "pitch" or "volume"). Instead, they represent abstract patterns learned from the audio signals.
and for the sake of better handling the train and test feature files for using it in lime and shap explainer i have converted and saved them in .npy and csv format where numpy files were used for lime and shap explainer code for feature contribution score graph representation
the files were saved as Train features: train_features.npy, Test Feature: test_features.npy.

3.Dummy Prediction Model
Since the final deep learning model is not yet available, a DummyModel class was created to simulate predictions. This dummy model generates random prediction scores for two classes (as a placeholder).
This is necessary for running the LIME and SHAP explainers, which require a model to provide predictions in order to explain them.

4.LIME (Local Interpretable Model-agnostic Explanations) was implemented to explain the contributions of each feature to the model’s predictions for individual instances.
The feature names are generated dynamically (as "Feature 1," "Feature 2," etc.) because the extracted features from wav2vec2 don’t have explicit labels.
The explainer was applied to a specific test instance to visualize the importance of features.

5.SHAP (SHapley Additive exPlanations) was implemented to explain the global and local contributions of features.
The SHAP explainer uses a KernelExplainer, which can explain predictions for any model. It was applied to a subset of 100 instances from the test set to limit memory usage.
SHAP summary and force plots were generated to visualize the feature contributions. Memory Optimization
To reduce memory consumption, the number of test samples used in the SHAP explainer is limited to 100 instances. This avoids overloading the memory when dealing with high-dimensional feature data.

6.Usage
LIME Explanation: The LIME explainer provides feature importance for a specific instance from the test set.
SHAP Explanation: SHAP is used for both local (per-instance) and global (across the test set) feature importance analysis. The force plot and summary plot show which features contribute positively or negatively to the model’s predictions.
