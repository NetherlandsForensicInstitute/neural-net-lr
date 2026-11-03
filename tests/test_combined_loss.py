from functools import partial

from DL_modules.criterion import CLLR
from DL_modules.criterion.base import CumulativeLoss, ProductLoss
from DL_modules.criterion.cllr import CllrTrainMode


def test_some_llrs_combined(cllr_instance_train_evenly, some_llrs_3_columns_tlm_result):
    tlm_result, labels = some_llrs_3_columns_tlm_result
    cllr_function = partial(CLLR, 1, CllrTrainMode.EVENLY)

    loss = cllr_function()(tlm_result, labels)
    alt_loss = [
        (1, CumulativeLoss([cllr_function()])),
        (1, ProductLoss([cllr_function()])),
        (2, CumulativeLoss([cllr_function()], weights=[2.0])),
        (2, CumulativeLoss([cllr_function(), cllr_function()])),
    ]

    for i, (factor, loss_function) in enumerate(alt_loss):
        assert loss * factor == loss_function(tlm_result, labels), f"comparison {i}"

    assert loss**2 == ProductLoss([cllr_function()], weights=[2.0])(tlm_result, labels)
    assert loss**2 == ProductLoss([cllr_function(), cllr_function()])(tlm_result, labels)
