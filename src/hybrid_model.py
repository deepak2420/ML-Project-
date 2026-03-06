import numpy as np


def hybrid_predict(rf, dl_model, X_test, X_test_dl):

    # Stage 1
    rf_predictions = rf.predict(X_test)

    # Find attack indices
    attack_indices = np.where(rf_predictions == 1)[0]

    # Stage 2 Deep Learning
    dl_predictions = dl_model.predict(
        X_test_dl[attack_indices],
        verbose=0
    )

    final_predictions = rf_predictions.copy()

    final_predictions[attack_indices] = (
        dl_predictions.flatten() > 0.5
    ).astype(int)

    return final_predictions