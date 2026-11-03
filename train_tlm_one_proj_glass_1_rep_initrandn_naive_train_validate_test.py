import confidence
import torch
from confidence import Configuration

from DL_modules.criterion import CLLR, CLLREvaluate
from DL_modules.criterion.cllr import CllrTrainMode
from DL_modules.custom_layers import TwoLevelNormalNormalOneProj
from DL_modules.dataloader import get_glass_1_rep_dataloaders_ttv
from DL_modules.train_validate_loops.pair_wise_llr_head import Track, train_and_validate_llr_head
from DL_modules.train_validate_loops.utils import get_next_entry

model_name = "tlm one rot, init randn, loss: cllr_train_random_projection"


def train_tlm_dl_model(cfg: Configuration):
    # get seed from config or from output folder
    seed = cfg.train.seed if cfg.train.seed else get_next_entry(cfg.saving.output_parent)
    # set seed for torch globally
    torch.manual_seed(seed)

    # Get dataloaders
    dataloader_train, dataloader_validate, dataloader_test = get_glass_1_rep_dataloaders_ttv(cfg)

    tlm_size = cfg.train.tlm_size
    # Define model
    top_layer = TwoLevelNormalNormalOneProj(
        tlm_size,
        cfg.training_data.n_same_source,
        variance_susp="randn",
        variance_cr="randn",
        variance_between="randn",
        n_axes_to_keep=cfg.train.n_axes_to_keep,
    )

    # model = PairToOutput(susp_net, cr_net, top_layer)
    model = top_layer

    # Define loss criterion and optimization parameters
    fr_hard_negs = cfg.training_data.fr_hard_negs
    criterion_train = CLLR(
        fr_hard_negs,
        CllrTrainMode(cfg.cllr_crit_train.mode),
        **cfg.cllr_crit_train.mode_options,
    )
    criterion_validate = CLLREvaluate(1)
    criterion_train_evaluate = CLLREvaluate(fr_hard_negs)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate)
    num_epochs = cfg.train.num_epochs

    coherence_filter = cfg.validate.coherence_filter
    coherence_filter_threshold_bias_axis = cfg.validate.coherence_filter_threshold_bias_axis
    coherence_filter_max_iterations = cfg.validate.coherence_filter_max_iterations
    coherence_filter_exclude_llrs_smaller_than = cfg.validate.coherence_filter_exclude_llrs_smaller_than

    input_for_criterion_evaluate = Track('t')

    train_and_validate_llr_head(
        model,
        model_name,
        criterion_train,
        criterion_validate,
        criterion_train_evaluate,
        num_epochs,
        optimizer,
        dataloader_train,
        dataloader_validate,
        cfg,
        tlm_size,
        input_for_criterion_evaluate,
        coherence_filter=coherence_filter,
        coherence_filter_threshold_bias_axis=coherence_filter_threshold_bias_axis,
        coherence_filter_max_iterations=coherence_filter_max_iterations,
        coherence_filter_exclude_llrs_smaller_than=coherence_filter_exclude_llrs_smaller_than,
    )


if __name__ == '__main__':
    cfg = confidence.load_name(
        'dl-data-params-glass-1-rep-train-validate-test',
        'train_tlm_one_proj_glass_1_rep_initrandn_naive_train_validate_test',
    )

    num_runs = cfg.trian.num_runs if cfg.train.num_runs else 1

    for _ in range(num_runs):
        train_tlm_dl_model(cfg)
