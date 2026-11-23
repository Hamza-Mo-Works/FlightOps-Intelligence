# Modeling methodology

The prediction interface estimates arrival delay risk using scheduled flight
information, historical rates and, for V2, historical weather snapshots. State
the target label from the training notebook when interpreting an evaluation.

Training, validation and test populations must be separated chronologically.
Fit imputers, categories, thresholds and feature statistics using only the
appropriate training/validation periods. Future arrival/departure outcomes,
actual elapsed times and target-month outcomes must not become predeparture
features. Prior-month rates must come from periods preceding the target flight.

The current V1 input builder reads a persisted route/month lookup and calendar.
Its UI inference period is November–December 2025. The V2 release notebook
reconstructs categorical levels from feature rows dated before 1 September 2025.
Those are observed source boundaries; verify the separate training/splitting
notebooks before claiming an end-to-end leakage audit.

Report ranking metrics, precision/recall at the chosen threshold, calibration,
class balance and population counts. A threshold is a decision rule, not a
probability estimate or a measure of accuracy. Historical evaluation is not
evidence that the same performance holds on future operational traffic.
