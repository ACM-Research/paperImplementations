![ACM Research Banner Light](https://github.com/ACM-Research/paperImplementations/assets/108421238/467a89e3-72db-41d7-9a25-51d2c589bfd9)

## Requirements

To install the necessary packages for this project, use the provided `requirements.txt` file. You can install the dependencies by running:

```pip install -r requirements.txt```

## Papers Read

1. A systematic comparison of deep learning methods for EEG time series analysis
2. Multi-disease Prediction Using LSTM Recurrent Neural Networks
3. Using recurrent neural network models for early detection of heart failure onset

## Paper 2 Chosen

**"Multi-disease Prediction Using LSTM Recurrent Neural Networks"**

## Summary of Paper

The paper "Multi-disease Prediction Using LSTM Recurrent Neural Networks" discusses the application of LSTM networks for predicting multiple diseases using EHR data. LSTMs are well-suited for this task due to their ability to maintain information over time and capture long-term dependencies in sequential data. The study demonstrates improved prediction accuracy for various medical conditions compared to traditional models.

## Justification for the Approach

The authors highlight the strengths of LSTMs in processing sequential and temporal data found in EHRs. Unlike traditional machine learning methods that may struggle with sequential dependencies, LSTMs excel by maintaining information over time through their gated architecture. This ability allows for more accurate modeling of patient history, leading to better prediction of disease onset and progression.

## Evaluation of Strengths and Weaknesses

**Strengths:**
- Demonstrates the effectiveness of LSTMs in handling sequential medical data.
- Captures temporal dependencies in patient histories, improving prediction accuracy.
- Applicable to a wide range of diseases, making the approach versatile.
- Outperforms traditional models that do not handle temporal data as effectively.

**Weaknesses:**
- Requires substantial computational resources for training.
- The complexity of the model may lead to overfitting if not properly regularized.
- LSTMs may struggle with very long sequences due to vanishing gradient issues, despite their gated architecture.
- Future research could explore optimizing model efficiency and extending applications to real-time disease monitoring and prediction.

## Some Novelties Noticed in the Paper

- **Sequential data processing:** The use of LSTM networks to handle sequential data in EHRs, capturing long-term dependencies that traditional models might miss.
- **Temporal dependency modeling:** The ability of LSTMs to maintain information over long sequences, leading to improved disease prediction.
- **Versatility in application:** The model's applicability to predicting multiple diseases from EHRs, showcasing its broad potential in healthcare.
- **Enhanced accuracy:** Demonstrated improvement in prediction accuracy over traditional models, validating the efficacy of LSTMs for this application.

