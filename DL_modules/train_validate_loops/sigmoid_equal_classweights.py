import copy
import logging
import os
from pathlib import Path

import lir
import numpy as np
import torch
import yaml
from confidence import Configuration
from torch.nn import Module
from torch.optim import Optimizer

from DL_modules.criterion.base import TlmLossFunction
from DL_modules.dataloader.data_loader import CustomDataLoader
from DL_modules.plots import calibration_plot, plot_losses
from DL_modules.train_validate_loops.utils import get_output_path
from DL_modules.utils import odds_to_logodds, probability_to_logodds, try_devpav

LOG = logging.getLogger(__name__)


def train_and_validate_sigmoid_equal_classweights_head(
    model: Module,
    model_name: str,
    criterion_train: TlmLossFunction,
    criterion_validate: Module,
    num_epochs: int,
    optimizer: Optimizer,
    dataloader_train: CustomDataLoader,
    dataloader_validate: CustomDataLoader,
    cfg: Configuration,
    output_dir_lr_benchmark: Path | None = None,
):
    if output_dir_lr_benchmark:
        output_path = f"{output_dir_lr_benchmark}/"
    elif cfg.saving.save:
        output_path = get_output_path(cfg.saving.output_parent)
    else:
        output_path = None

    if output_path:
        # save configuation
        os.makedirs(Path(output_path), exist_ok=True)

    # Initialize loss tracking
    train_loss = []
    validate_loss = []
    model = model.double()
    # Start training
    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        total_train_loss = 0.0
        num_batches = len(dataloader_train)

        for batch_idx, (inputs) in enumerate(dataloader_train):
            optimizer.zero_grad()
            susp, cr, labels = inputs
            # moving tensors to GPU does not speed things up a lot. Probably since the model is so simple.
            # susp.to('cuda')
            # cr.to('cuda')
            # labels.to('cuda')
            predictions = model(susp.double(), cr.double())
            loss = criterion_train(predictions, labels.float())  # Convert labels to float
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            # Show progress
            if (batch_idx + 1) % 10 == 0:
                LOG.debug(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx}/{num_batches}]")
        if (epoch + 1) % 10 == 0:
            calibration_plot(
                probability_to_logodds(predictions[:, 0].detach().numpy()),
                labels.numpy(),
                title=f"PAV-plot train, epoch = {epoch}",
            )

        # Keep track of train losses per epoch
        average_train_loss = total_train_loss / num_batches
        train_loss.append(average_train_loss)

        # Compute validate loss
        model.eval()
        all_predictions = torch.empty(0, 1)
        all_labels = torch.empty(0)
        with torch.no_grad():
            for all_data in dataloader_validate:
                susp, cr, labels = all_data
                predictions = model(susp.double(), cr.double())
                all_predictions = torch.cat((all_predictions, predictions))
                all_labels = torch.cat((all_labels, labels))
        current_validate_loss = criterion_validate(all_predictions, all_labels.float())  # Convert labels to float

        # Keep track of validate losses
        validate_loss.append(current_validate_loss.item())

        # keep track of train losses
        all_predictions_train = torch.empty(0, 1)
        all_labels_train = torch.empty(0)
        with torch.no_grad():
            for all_data in dataloader_train:
                susp, cr, labels = all_data
                predictions = model(susp.double(), cr.double())
                all_predictions_train = torch.cat((all_predictions_train, predictions))
                all_labels_train = torch.cat((all_labels_train, labels))
        #        current_train_loss_on_validate_crit = criterion_train(all_tlmresults_train, all_labels_train.float())

        # Keep track of best
        if epoch == 0 or float(current_validate_loss) < np.min(validate_loss[:-1]):
            best_model = copy.deepcopy(model)
            best_epoch = epoch
            train_loss_with_best = average_train_loss
            best_validate_loss = current_validate_loss
            best_LRs_validate = (all_predictions[:, 0] / (1 - all_predictions[:, 0])).numpy()
            best_llrs_validate = odds_to_logodds(best_LRs_validate)
            best_labels_validate = all_labels.numpy()
            best_LRs_train = (all_predictions_train[:, 0] / (1 - all_predictions_train[:, 0])).numpy()
            best_labels_train = all_labels_train.numpy()
            devpav_best_validate = try_devpav(best_LRs_validate, best_labels_validate)
            LOG.debug(f"devpav_best_validate {devpav_best_validate}")

        # Show progress
        LOG.debug(
            f"Epoch [{epoch}/{num_epochs}], Average Train Loss: {average_train_loss:.4f},"
            f"Validate Loss: {current_validate_loss.item():.4f}"
        )

        if (epoch + 1) % 10 == 0:
            # Show calibration plot of validate data every tenth epoch
            calibration_plot(
                probability_to_logodds(all_predictions[:, 0].numpy()),
                all_labels.numpy(),
                title=f"PAV-plot validate, epoch = {epoch}",
            )
            # Save calibration plots of train and validate at end of every 10th epoch
            if cfg.saving.save:
                os.makedirs(Path(output_path), exist_ok=True)
                calibration_plot(
                    probability_to_logodds(predictions[:, 0].numpy()),
                    labels.numpy(),
                    title=f"PAV-plot train, epoch = {epoch}",
                    show_plot=False,
                    pathfile_save=f"{output_path}pav_plot_train_epoch_{epoch}_batch_{batch_idx + 1}.png",
                )
                calibration_plot(
                    probability_to_logodds(all_predictions[:, 0].numpy()),
                    all_labels.numpy(),
                    title=f"PAV-plot validate, epoch = {epoch}",
                    show_plot=False,
                    pathfile_save=f"{output_path}pav_plot_validate_epoch_{epoch}.png",
                )

    if cfg.saving.save:
        torch.save(best_model, Path(output_path) / f"{model_name}.pth")

        # apply best model to validation dataset
        # Compute validation loss
        # best_model_loaded = torch.load(Path(output_path) / f'{model_name}.pth')

        # Plot losses, and save train and validation results for best epoch
        plot_losses(
            num_epochs,
            train_loss,
            validate_loss,
            pathfile_save=f"{output_path}loss_plot.png",
        )
        with lir.plotting.show() as ax:
            ax.ece(best_LRs_validate, best_labels_validate)
            ax.savefig(output_path + f"best_ece_epoch={best_epoch}.png")
        with lir.plotting.show() as ax:
            ax.ece(best_LRs_validate, best_labels_validate)
            ax.ylim(0, 2.0 * float(best_validate_loss))
            ax.savefig(output_path + f"best_ece_epoch={best_epoch}_zoom.png")
        with lir.plotting.show() as ax:
            ax.lr_histogram(np.clip(best_LRs_validate, 10**-15, 10**15), best_labels_validate)
            ax.xlim(-6, 6)
            ax.savefig(output_path + f"best_llr_histogram_validate_epoch={best_epoch}_validate.png")

        best_llrs_train = odds_to_logodds(best_LRs_train)

        calibration_plot(
            best_llrs_train,
            best_labels_train,
            title=f"PAV-plot train, epoch = {best_epoch}",
            show_plot=False,
            pathfile_save=f"{output_path}train_at_best_validate_pav_plot_epoch_{best_epoch}.png",
        )
        calibration_plot(
            best_llrs_validate,
            best_labels_validate,
            title=f"PAV-plot validate, epoch = {best_epoch}",
            show_plot=False,
            pathfile_save=f"{output_path}best_validate_pav_plot_epoch_{best_epoch}.png",
        )

        # Save training results to a yaml file
        output_params = {
            "model": model_name,
            "best_epoch": best_epoch,
            "best_cllr_validate": float(best_validate_loss),
            "accompanying_train_loss": float(train_loss_with_best),
            "train_losses": train_loss,
            "validate_losses": validate_loss,
            "train_ss_batch_size": cfg.train.ss_batch_size,
            "validate_ss_batch_size": cfg.validate.ss_batch_size,
            "test_ss_batch_size": cfg.test.ss_batch_size,
            "devpav_best_validate": float(devpav_best_validate),
        }
        with open(f"{output_path}results.yaml", "w") as file:
            yaml.dump(output_params, file, default_flow_style=False)
    else:
        plot_losses(num_epochs, train_loss, validate_loss)

    # After training completes, print train_loss and validate_loss and best
    LOG.debug(f"Train Losses: {train_loss}")
    LOG.debug(f"Validate Losses: {validate_loss}")
    LOG.debug(
        f"Best validate loss at {float(best_validate_loss):.4f}, and accompanying train loss at "
        f"{float(train_loss_with_best):.4f} at epoch {best_epoch}."
    )

    return best_model
