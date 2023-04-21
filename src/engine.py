import numpy as np
from src.builders import (
    dataloader_builder,
    model_builder,
    transform_builder,
    criterion_builder,
    scheduler_builder,
    optimizer_builder,
    evaluator_builder,
    meter_builder,
)
from src.utils.misc import set_random_seed
import torch.nn as nn
import torch
import os
import time
from tqdm import tqdm
import pandas as pd
from attributedict.collections import AttributeDict
import json


try:
    import wandb
except ImportError:
    print("Install wandb if cloud model monitoring is required.")


class Engine:
    def __init__(
        self, config, save_dir, logger, train=True, sweep=False, pretrain=True
    ):

        self.logger = logger

        # Where to save trained model and other artefacts
        self.save_dir = save_dir

        if not sweep:
            # Log the config
            self.logger.info(json.dumps(config))

        config = AttributeDict(config)

        # Set the seed
        set_random_seed(config.train.seed)

        # Init wandb config beforehand if sweeping
        if sweep:
            self._init_wandb(config=config)
            config = self._modify_config_for_sweep(config)

        # # Process config
        self._process_configs(config=config, train=train, sweep=sweep)

        # Initialize wandb
        if self.config.train.use_wandb and not sweep:
            self._init_wandb(config=config)

        # Keep the best metric value
        if self.config.train.evaluator.maximize:
            self.best_eval_metric = float("-inf")
        else:
            self.best_eval_metric = float("inf")

        # Other required variables
        self.train_num_steps = 0
        self.training_epoch_steps = None
        self.val_num_steps = None
        self.val_epoch_steps = None
        self.pretrain = pretrain

    @staticmethod
    def _modify_config_for_sweep(config):
        sweep_config = wandb.config

        # Train configs
        for (key, val) in sweep_config["train"]["optimizer"].items():
            config.train.optimizer[key] = val

        for (key, val) in sweep_config["train"]["criterion"].items():
            config.train.criterion[key] = val

        # Model configs
        for (key, val) in sweep_config["model"].items():
            config.model[key] = val

        # Data configs
        # for (key, val) in sweep_config["data"].items():
        #     config.data[key] = val
        frames_config = sweep_config["data"]["max_frames_sampled_frames"]
        config.data["max_frames"] = frames_config[0]
        config.data["n_sampled_frames"] = frames_config[1]

        return config

    @staticmethod
    def _init_wandb(config):
        wandb.init(
            project=config.train.wandb_project_name,
            name=config.train.wandb_run_name,
            config=config,
            entity=config.train.wandb_entity,
            group=config.train.wandb_group_name,
            mode=config.train.wandb_mode,
        )

        # define our custom x axis metric
        wandb.define_metric("batch_train/step")
        wandb.define_metric("batch_valid/step")
        wandb.define_metric("epoch")

        # set all other metrics to use the corresponding step
        wandb.define_metric("batch_train/*", step_metric="batch_train/step")
        wandb.define_metric("batch_valid/*", step_metric="batch_valid/step")
        wandb.define_metric("epoch/*", step_metric="epoch")
        wandb.define_metric("lr", step_metric="epoch")
        wandb.define_metric("training_throughput", step_metric="epoch")
        wandb.define_metric("time_per_step", step_metric="epoch")

    def _process_configs(self, config, train, sweep):
        # Useful flags used by the engine
        self.train = train
        self.sweep = sweep
        self.config = config

        self.config.model.update({"mode": self.config.train.mode})
        # Make sure wandb_log_steps is divisble by batch size
        self.config.train.batch_size = (
            self.config.train.batch_size if self.config.train else 1
        )
        self.config.train.wandb_log_steps = self.config.train.wandb_log_steps + (
            self.config.train.batch_size
            - self.config.train.wandb_log_steps % self.config.train.batch_size
        )
        # Process optimizer and scheduler configs
        self.config.train.optimizer.update({"mode": self.config.train.mode})
        self.config.train.scheduler.update(
            {
                "mode": self.config.train.mode,
                "epochs": self.config.train.epochs,
                "batch_size": self.config.train.batch_size,
            }
        )
        self.config.data.update(
            {"mode": self.config.train.mode,
             "batch_size": self.config.train.batch_size})

    def _build(self):

        # Build the required torchvision transforms
        self.transform, self.aug_transform = transform_builder.build(self.config.data)

        # Build the dataloaders
        self.dataloaders, self.data_dirs = dataloader_builder.build(
            self.config.data,
            self.train,
            self.transform,
            self.aug_transform,
            logger=self.logger,
        )

        # Build the model
        self.checkpoint_path = self.config.model.pop("checkpoint_path")
        # Add the batch and num_frame info to the model backbone config
        self.config.model["backbone"].update({"num_frames": self.config.data.max_frames,
                                              # Hacky addition of num_vids to the backbone config
                                              "num_vids": 2})
        self.model = model_builder.build(self.config.model)

        # Build the criteria
        self.criterion = criterion_builder.build(
            self.config.train.criterion, self.config.train.mode
        )
        self.loss_meters = meter_builder.build()

        # Build the evaluator
        self.evaluator = evaluator_builder.build(self.config.train.evaluator)

        # Build the optimizer and the scheduler
        if self.train:
            self.optimizer = optimizer_builder.build(
                self.model, self.config.train.optimizer
            )
            self.scheduler = scheduler_builder.build(
                self.optimizer, self.config.train.scheduler
            )

        # Load the checkpoint
        self.misc = self.load_checkpoint()

        # Move model to correct device
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.model = nn.DataParallel(self.model).to(self.device)

            # Pretrained models?
            if self.train:
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()


    def train_model(self):

        self._build()

        # Number of iterations per epoch
        self.training_epoch_steps = len(self.dataloaders["train"])
        self.val_epoch_steps = len(self.dataloaders["val"])

        # Starting epoch
        start_epoch = 0 if not self.misc else self.misc["epoch"]

        for epoch in range(start_epoch, start_epoch + self.config.train.epochs):

            # Reset the evaluators
            self.reset_evaluator()
            self.reset_meters()

            # Start timer
            start_time = time.time()

            # Start training for one epoch
            self._train_one_epoch(epoch)

            total_time = time.time() - start_time

            # Compute training throughput and time per step
            self.log_computation_stats(epoch=epoch, total_time=total_time)

            self.val_num_steps = self.train_num_steps
            self.reset_evaluator()
            self.reset_meters()

            # Perform validation
            self._evaluate_one_epoch(epoch, True)

    def log_computation_stats(self, epoch, total_time):
        training_throughput = (
            self.config.train.batch_size * len(self.dataloaders["train"]) / total_time
        )
        time_per_step = self.config.train.batch_size / training_throughput

        self.logger.info(
            "Epoch finished with a training throughput of {} and time per step of {}".format(
                training_throughput, time_per_step
            )
        )

        if self.config.train.use_wandb:
            wandb.log(
                {
                    "training_throughput": training_throughput,
                    "time_per_step": time_per_step,
                    "epoch": epoch,
                }
            )

    def evaluate(self):
        self._build()

        self.val_epoch_steps = len(self.dataloaders["val"])
        self.val_num_steps = self.train_num_steps

        self._evaluate_one_epoch(0, False)

    def _prepare_input_data(self, data_iter):
        data_dict = next(data_iter)

        for key in data_dict.keys():
            if data_dict[key] is not None:
                data_dict[key] = data_dict[key].to(self.device)

        return data_dict

    def _train_one_epoch(self, epoch):

        # Change to train mode
        self.model.train()

        # Initialize TQDM iterator
        iterator = tqdm(range(len(self.dataloaders["train"])), dynamic_ncols=True)
        data_iter = iter(self.dataloaders["train"])

        for i in iterator:

            data_dict = self._prepare_input_data(data_iter)

            self.optimizer.zero_grad()

            (
                loss,
                output,
            ) = self._forward_path(data_dict)

            
            # Backward prop
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                # Update evaluators
                self.update_evaluator(
                    output.detach().cpu().numpy(),
                    data_dict["label"].detach().cpu().numpy(),
                )
                # Update TQDM bar
                self._set_tqdm_description(
                    iterator=iterator,
                    log_mode="Training",
                    epoch=epoch,
                    loss=self.loss_meters["main_loss"].avg,
                )

                self.train_num_steps += self.config.train.batch_size

        self._train_epoch_summary(epoch=epoch, lr=self.optimizer.param_groups[0]["lr"])

    def _evaluate_one_epoch(self, epoch, train=True):

        with torch.no_grad():

            # logits = list()

            # Change to eval mode
            self.model.eval()

            # Initialize TQDM iterator
            iterator = tqdm(range(len(self.dataloaders["val"])), dynamic_ncols=True)
            data_iter = iter(self.dataloaders["val"])

            for i in iterator:
                data_dict = self._prepare_input_data(data_iter)

                # In validation the video shape is (batch_size, num_views, num_videos, num_frames, , height, width, channels)
                # We need (batch_size, num_videos, num_frames, height, width, channels)
                data_dict["vid"] = data_dict["vid"].squeeze(0)
                data_dict["mask"] = data_dict["mask"].squeeze(0)

                y_pred = self.model(data_dict)

                output = torch.mean(y_pred, dim=0)

                # if not train:
                #     logits.append(output.detach().cpu().numpy())

                # Compute loss
                loss = self.criterion["classification"](output.unsqueeze(0), data_dict["label"])

                with torch.no_grad():
                    self.loss_meters["main_loss"].update(loss.detach().cpu().item())

                self._set_tqdm_description(
                    iterator=iterator,
                    log_mode="Validation" if self.train else "Test",
                    epoch=epoch,
                    loss=self.loss_meters["main_loss"].avg,
                )

                self.update_evaluator(
                    output.unsqueeze(0).detach().cpu().numpy(),
                    data_dict["label"].detach().cpu().numpy(),
                )

                self.val_num_steps += 1

            eval_metric = self.evaluator[
                self.config.train.evaluator.eval_metric
            ].compute()

            if train:
                self.scheduler.step(eval_metric)

            if self.train:
                if self.config.train.evaluator.maximize:
                    self.best_eval_metric = (
                        eval_metric
                        if eval_metric > self.best_eval_metric
                        else self.best_eval_metric
                    )
                else:
                    self.best_eval_metric = (
                        eval_metric
                        if eval_metric < self.best_eval_metric
                        else self.best_eval_metric
                    )

                if not self.sweep:
                    self.save_checkpoint(epoch=epoch, eval_metric=eval_metric)

                self.log_wandb_summary()
            else:
                # Save predictions csv
                prediction_df = pd.DataFrame(
                    {
                        "preds": self.evaluator[
                            self.config.train.evaluator.eval_metric
                        ].y_pred,
                        "labels": self.evaluator[
                            self.config.train.evaluator.eval_metric
                        ].y_true,
                        # "logits": logits,
                    }
                )
                prediction_df.to_csv(os.path.join(self.save_dir, "preds.csv"))

            self._val_epoch_summary(epoch=epoch)

    def _train_epoch_summary(
        self,
        epoch,
        lr,
    ):

        log_str = "Training Epoch {} - Total Loss = {} ".format(
            epoch, self.loss_meters["main_loss"].avg
        )

        if self.config.train.use_wandb:
            wandb.log(
                {
                    "epoch/train_total_loss": self.loss_meters["main_loss"].avg,
                    "epoch": epoch,
                }
            )

        log_str += "- LR = {} ".format(
            lr,
        )

        if self.config.train.use_wandb:
            wandb.log({"lr": lr, "epoch": epoch})


        for eval_type in self.evaluator.keys():
            log_str += (
                "- "
                + eval_type.upper()
                + " = {}".format(self.evaluator[eval_type].compute())
            )

            if self.config.train.use_wandb:
                wandb.log(
                    {
                        "epoch/train_{}_score".format(eval_type): self.evaluator[
                            eval_type
                        ].compute(),
                        "epoch": epoch,
                    }
                )

        self.logger.info(log_str)

    def _val_epoch_summary(self, epoch):

        log_str = "{} Epoch {} - Main Loss = {} ".format(
            "Validation" if self.train else "Test",
            epoch,
            self.loss_meters["main_loss"].avg,
        )

        if self.config.train.use_wandb:
            wandb.log(
                {
                    "epoch/val_main_loss": self.loss_meters["main_loss"].avg,
                    "epoch": epoch,
                }
            )

        for eval_type in self.evaluator.keys():
            log_str += (
                "- "
                + eval_type.upper()
                + " = {} ".format(self.evaluator[eval_type].compute())
            )

            if self.config.train.use_wandb:
                wandb.log(
                    {
                        "epoch/val_{}_score".format(eval_type): self.evaluator[
                            eval_type
                        ].compute(),
                        "epoch": epoch,
                    }
                )


            if self.config.train.use_wandb:
                wandb.log(
                    {
                        "epoch/best_eval_metric".format(
                            eval_type
                        ): self.best_eval_metric,
                        "epoch": epoch,
                    }
                )
        log_str += "- Best {} Measurement = {}".format(
                eval_type.upper(), self.best_eval_metric
            )
        self.logger.info(log_str)

    def _forward_path(self, data_dict):
        output_dict = self.model(data_dict)

        # Compute loss
        loss = self.criterion["classification"](output_dict, data_dict["label"])
        
        with torch.no_grad():
            self.loss_meters["main_loss"].update(loss.detach().cpu().item())

        # TODO: Should I add other losses and attention here
        return loss, output_dict
    
    @staticmethod
    def _set_tqdm_description(iterator, log_mode, epoch, loss):

        iterator.set_description(
            "[Epoch {}] | {} | Loss: {:.4f}".format(epoch, log_mode, loss),
            refresh=True,
        )

    def log_wandb_summary(self):
        if self.config.train.use_wandb:
            wandb.run.summary["best_eval_metric"] = self.best_eval_metric

    # def is_rank_0(self):
    #     return get_ddp_save_flag() or not self.ddp

    def reset_evaluator(self):
        for key in self.evaluator.keys():
            self.evaluator[key].reset()

    def reset_meters(self):
        for key in self.loss_meters.keys():
            self.loss_meters[key].reset()

    def update_evaluator(self, pred, label):
        for key in self.evaluator.keys():
            self.evaluator[key].update(y_pred=pred, y_true=label)

    def save_checkpoint(self, epoch, eval_metric):

        checkpoint = {
            "epoch": epoch,
            "best_eval_metric": self.best_eval_metric,
            "eval_metric": eval_metric,
        }

        for eval_type in self.evaluator.keys():
            checkpoint["metric_{}".format(eval_type)] = self.evaluator[
                eval_type
            ].compute()

        # Add model state dicts
        try:
            checkpoint["model"] = self.model.module.state_dict()
        except AttributeError:
            checkpoint.model = self.model.state_dict()

        # Add optimizer state dicts
        checkpoint["optimizer"] = self.optimizer.state_dict()

        # Add scheduler state dicts
        checkpoint["scheduler"] = self.scheduler.state_dict()

        # Save last_checkpoint
        checkpoint_path = os.path.join(self.save_dir, "checkpoint_last.pth")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info("last_checkpoint is saved for epoch {}.".format(epoch))

        # Save the best_checkpoint if performance improved
        if eval_metric != self.best_eval_metric:
            return

        # Save best_checkpoint
        checkpoint_path = os.path.join(self.save_dir, "checkpoint_best.pth")
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(
            "best_checkpoint is saved for epoch {} with eval metric {}.".format(
                epoch, eval_metric
            )
        )

    def load_checkpoint(self):
        checkpoint = None
        try:
            if self.checkpoint_path:
                self.logger.info("Loading checkpoint from {}".format(self.checkpoint_path))
                checkpoint = torch.load(self.checkpoint_path)

                # Load model weights
                self.model.load_state_dict(checkpoint.pop("model"), strict=False)

                if self.train:
                    # Load optimizer state
                    self.optimizer.load_state_dict(checkpoint.pop("optimizer"))

                    # Load scheduler state
                    self.scheduler.load_state_dict(checkpoint.pop("scheduler"))
        except Exception as e:
            self.logger.error("Failed to load checkpoint: {}".format(e))
        finally:
            return checkpoint
