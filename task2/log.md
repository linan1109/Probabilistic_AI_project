##### No std, only theta_bar
        inference_mode: InferenceMode = InferenceMode.SWAG_DIAGONAL,
        swag_epochs: int = 30,
        swag_learning_rate: float = 0.045,
        swag_update_freq: int = 3,
        deviation_matrix_max_rank: int = 15,
        bma_samples: int = 1,
        self._prediction_threshold = 2.0 / 3.0
Accuracy (raw): 0.6786
Accuracy (non-ambiguous only, your predictions): 0.7167
Accuracy (non-ambiguous only, predicting most-likely class): 0.8667
Best cost 0.5642856955528259 at threshold 0.6776280403137207
Note that this threshold does not necessarily generalize to the test set!
Validation ECE: 0.12293880198683059
Your PUBLIC test cost is 0.929 and PUBLIC test ECE is 0.220.
Your penalized PUBLIC cost 1.050 is thus worse than the easy baseline's PUBLIC cost 0.901.

##### with std, no D
        inference_mode: InferenceMode = InferenceMode.SWAG_DIAGONAL,
        swag_epochs: int = 30,
        swag_learning_rate: float = 0.045,
        swag_update_freq: int = 3,
        deviation_matrix_max_rank: int = 15,
        bma_samples: int = 30,
        self._prediction_threshold = 2.0 / 3.0
Accuracy (raw): 0.6214
Accuracy (non-ambiguous only, your predictions): 0.6333
Accuracy (non-ambiguous only, predicting most-likely class): 0.8667
Best cost 0.6071428656578064 at threshold 0.6874220371246338
Validation ECE: 0.08276821609054293
Your PUBLIC test cost is 0.888 and PUBLIC test ECE is 0.136.
Your penalized PUBLIC cost 0.924 is thus worse than the easy baseline's PUBLIC cost 0.901.

##### with std, with D
        inference_mode: InferenceMode = InferenceMode.SWAG_FULL,
        swag_epochs: int = 30,
        swag_learning_rate: float = 0.045,
        swag_update_freq: int = 3,
        deviation_matrix_max_rank: int = 15,
        bma_samples: int = 30,
        self._prediction_threshold = 2.0 / 3.0
Accuracy (raw): 0.6571
Accuracy (non-ambiguous only, your predictions): 0.6833
Accuracy (non-ambiguous only, predicting most-likely class): 0.8583
Best cost 0.5785714387893677 at threshold 0.753808856010437
Note that this threshold does not necessarily generalize to the test set!
Validation ECE: 0.09246459880045484
Your PUBLIC test cost is 0.902 and PUBLIC test ECE is 0.151.
Your penalized PUBLIC cost 0.953 is thus worse than the easy baseline's PUBLIC cost 0.901.

##### with std, with D, threshold 0.75
        inference_mode: InferenceMode = InferenceMode.SWAG_FULL,
        swag_epochs: int = 30,
        swag_learning_rate: float = 0.045,
        swag_update_freq: int = 3,
        deviation_matrix_max_rank: int = 15,
        bma_samples: int = 30,
        self._prediction_threshold = 0.75
Accuracy (raw): 0.6214
Accuracy (non-ambiguous only, your predictions): 0.6250
Accuracy (non-ambiguous only, predicting most-likely class): 0.8583
Best cost 0.5785714387893677 at threshold 0.753808856010437

Validation ECE: 0.09246459880045484
Your PUBLIC test cost is 0.858 and PUBLIC test ECE is 0.151.
Your penalized PUBLIC cost 0.909 is thus worse than the easy baseline's PUBLIC cost 0.901.

#### more steps
        inference_mode: InferenceMode = InferenceMode.SWAG_FULL,
        swag_epochs: int = 100,
        swag_learning_rate: float = 0.045,
        swag_update_freq: int = 5,
        deviation_matrix_max_rank: int = 15,
        bma_samples: int = 100,
        self._prediction_threshold = 0.75
Accuracy (raw): 0.6286
Accuracy (non-ambiguous only, your predictions): 0.5917
Accuracy (non-ambiguous only, predicting most-likely class): 0.9250
Best cost 0.550000011920929 at threshold 0.0
Validation ECE: 0.10425598749092646
Your PUBLIC test cost is 0.842 and PUBLIC test ECE is 0.075.
Your penalized PUBLIC cost 0.842 is thus worse than the hard baseline's PUBLIC cost 0.841.

#### more steps and threshold = 0.25
        inference_mode: InferenceMode = InferenceMode.SWAG_FULL,
        swag_epochs: int = 100,
        swag_learning_rate: float = 0.045,
        swag_update_freq: int = 5,
        deviation_matrix_max_rank: int = 15,
        bma_samples: int = 100,
        self._prediction_threshold = 0.25
Accuracy (raw): 0.7929
Accuracy (non-ambiguous only, your predictions): 0.9250
Accuracy (non-ambiguous only, predicting most-likely class): 0.9250
Best cost 0.4571428596973419 at threshold 0.575624942779541
Validation ECE: 0.10425598749092646
Your PUBLIC test cost is 1.252 and PUBLIC test ECE is 0.075.
Your penalized PUBLIC cost 1.252 is thus worse than the easy baseline's PUBLIC cost 0.901.

#### more steps and threshold = 0.6
        inference_mode: InferenceMode = InferenceMode.SWAG_FULL,
        swag_epochs: int = 100,
        swag_learning_rate: float = 0.045,
        swag_update_freq: int = 5,
        deviation_matrix_max_rank: int = 15,
        bma_samples: int = 100,
        self._prediction_threshold = 0.6
Your PUBLIC test cost is 0.863 and PUBLIC test ECE is 0.075.
Your penalized PUBLIC cost 0.863 is thus worse than the medium baseline's PUBLIC cost 0.856.


#### add dynamic threshold
        inference_mode: InferenceMode = InferenceMode.SWAG_FULL,
        swag_epochs: int = 100,
        swag_learning_rate: float = 0.045,
        swag_update_freq: int = 5,
        deviation_matrix_max_rank: int = 15,
        bma_samples: int = 100,
threshold 0.5765765765765766 for cost 0.45714286
Evaluating model on validation data
Accuracy (raw): 0.7714
Accuracy (non-ambiguous only, your predictions): 0.7917
Accuracy (non-ambiguous only, predicting most-likely class): 0.9250
Best cost 0.4642857015132904 at threshold 0.6086249351501465
Note that this threshold does not necessarily generalize to the test set!
Validation ECE: 0.11598784114633286
Congratulations, you beat the easy baseline in terms of public cost.
However, you did not pass the medium baseline yet.
Your PUBLIC test cost is 0.873 and PUBLIC test ECE is 0.086.
Your penalized PUBLIC cost 0.873 is thus worse than the medium baseline's PUBLIC cost 0.856.

### 0.7 threshold

        swag_epochs: int = 50,
        swag_learning_rate: float = 0.045,
        swag_update_freq: int = 1,
        deviation_matrix_max_rank: int = 15,
        bma_samples: int = 30,


Accuracy (raw): 0.5571
Accuracy (non-ambiguous only, your predictions): 0.5333
Accuracy (non-ambiguous only, predicting most-likely class): 0.8917
Best cost 0.6357142925262451 at threshold 0.7015841007232666
Note that this threshold does not necessarily generalize to the test set!
Validation ECE: 0.12843042420489445
Predicting probabilities on test data
Obtaining labels from your predictions

Congratulations, you beat the hard baseline in terms of public cost.
Your PUBLIC test cost is 0.833 and PUBLIC test ECE is 0.075.
Your penalized PUBLIC cost is 0.833.

###

        swag_epochs: int = 100,
        swag_learning_rate: float = 0.045,
        swag_update_freq: int = 5,
        deviation_matrix_max_rank: int = 15,
        bma_samples: int = 100,
        THRESHOLD = 0.7

Evaluating model on validation data
Accuracy (raw): 0.7000
Accuracy (non-ambiguous only, your predictions): 0.7000
Accuracy (non-ambiguous only, predicting most-likely class): 0.9000
Best cost 0.5142857432365417 at threshold 0.7021021246910095
Note that this threshold does not necessarily generalize to the test set!
Validation ECE: 0.09580733541931424
Predicting probabilities on test data
Obtaining labels from your predictions

Congratulations, you beat the hard baseline in terms of public cost.
Your PUBLIC test cost is 0.818 and PUBLIC test ECE is 0.109.
Your penalized PUBLIC cost is 0.827.

### 

        swag_epochs: int = 100,
        swag_learning_rate: float = 0.03,
        swag_update_freq: int = 2,
        deviation_matrix_max_rank: int = 45,
        bma_samples: int = 100,

Running your solution
Loaded pretrained MAP weights from map_weights.pt
Evaluating model on validation data
Accuracy (raw): 0.7143
Accuracy (non-ambiguous only, your predictions): 0.7167
Accuracy (non-ambiguous only, predicting most-likely class): 0.8833
Best cost 0.5285714268684387 at threshold 0.0
Note that this threshold does not necessarily generalize to the test set!
Validation ECE: 0.09845355706555502
Predicting probabilities on test data
Obtaining labels from your predictions

Congratulations, you beat the easy baseline in terms of public cost.
However, you did not pass the medium baseline yet.
Your PUBLIC test cost is 0.832 and PUBLIC test ECE is 0.163.
Your penalized PUBLIC cost 0.895 is thus worse than the medium baseline's PUBLIC cost 0.856.
Dumped check file to results_check.byte

### 

        inference_mode: InferenceMode = InferenceMode.SWAG_FULL,
        swag_epochs: int = 100,
        swag_learning_rate: float = 0.045,
        swag_update_freq: int = 10,
        deviation_matrix_max_rank: int = 5,
        bma_samples: int = 100,

Accuracy (raw): 0.6500
Accuracy (non-ambiguous only, your predictions): 0.6333
Accuracy (non-ambiguous only, predicting most-likely class): 0.9000
Best cost 0.5428571701049805 at threshold 0.7537959218025208
Note that this threshold does not necessarily generalize to the test set!
Validation ECE: 0.108403612353972
Predicting probabilities on test data
Obtaining labels from your predictions

Congratulations, you beat the easy baseline in terms of public cost.
However, you did not pass the medium baseline yet.
Your PUBLIC test cost is 0.850 and PUBLIC test ECE is 0.111.
Your penalized PUBLIC cost 0.861 is thus worse than the medium baseline's PUBLIC cost 0.856.
Dumped check file to results_check.byte
