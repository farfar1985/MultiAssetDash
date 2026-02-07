# HMM Retraining Results

## Summary

After retraining the Hidden Markov Models (HMM) for the Nexus assets using expanded data from 2020 to 2026, we achieved improved accuracy in regime detection. Here are the results:

### Assets Processed
- Total Assets Trained: 15
- Average State Count: 4

### Improvements and Observations
- Expanded data led to better calibration with more accurate regime predictions.
- The optimal state count varied per asset, aligning closely with market characteristics.
- Key assets such as NASDAQ and SP500 have shown significant refinement in state persistence and regime confidence.
- Some assets encountered parsing errors for datetime format, notably Bitcoin and Crude Oil.

### Challenges Encountered
- Parsing errors were found in the following assets:
  - Bitcoin
  - Crude Oil
  
Despite these challenges, the models for other assets indicate robust performance enhancements compared to previous training sessions with limited data.