from lrbenchmark.pairing import InstancePairing

from DL_modules.criterion import CLLR
from DL_modules.lir_extensions.two_level_nn import TwoLevelSystemPairNN, TwoLevelSystemSingleInstanceNN


def test_two_level_pair_nn(tmp_path, synthetic_data_labels):
    features, labels = synthetic_data_labels
    lrsystem = TwoLevelSystemPairNN(
        "nn",
        preprocessing_pipeline=None,
        pairing_function=InstancePairing(),
        loss_function=CLLR(1, "RANDOM_PROJECTION"),
        ss_batch_size_train=5,
        ss_batch_size_validate=5,
        learning_rate=0.003,
        num_epochs=5,
        show_plots=False,
        output_dir=tmp_path,
        axes_to_monitor="all",
        tlm_size=features.shape[1],
        n_axes_to_keep=features.shape[1],
        tlm_pair_version="single_projection",
        threshold_for_mean_cllr=0.95,
    )
    lrsystem.fit(features, labels, labels)
    lrsystem.apply(features, labels, labels)


def test_two_level_single_instance_nn(tmp_path, synthetic_data_labels):
    features, labels = synthetic_data_labels
    lrsystem = TwoLevelSystemSingleInstanceNN(
        "nn",
        preprocessing_pipeline=None,
        pairing_function=InstancePairing(),
        loss_function=CLLR(1, "RANDOM_PROJECTION"),
        ss_batch_size_train=5,
        ss_batch_size_validate=5,
        learning_rate=0.003,
        num_epochs=5,
        show_plots=False,
        output_dir=tmp_path,
        axes_to_monitor="all",
        tlm_size=features.shape[1],
        n_axes_to_keep=features.shape[1],
        tlm_single_instance_version='single_projection_normal',
        threshold_for_mean_cllr=0.95,
        also_predict_specific_source=True,
        seed=1,
    )

    lrsystem.fit(features, labels, labels)
    lrsystem.apply(features, labels, labels)
