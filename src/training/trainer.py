"""
Pico Language Model Trainer

This Trainer implements a minimalistic end-to-end training pipeline of the Pico language model with
distributed training support via Lightning Fabric. It provides a modular and configurable training
pipeline with the features:

    - Configuration Management: YAML-based configuration for all aspects of training
    - Distributed Training: Multi-GPU support via Lightning Fabric
    - Checkpointing: Regular model saving and training state recovery
    - Evaluation: Periodic model evaluation on validation datasets
    - Logging: Comprehensive metric tracking and experiment monitoring
    - Optimization: Support for gradient accumulation, clipping, and LR scheduling
"""

import logging
import lightning as L
import torch
import torch.nn.functional as F
import random
import os
from lightning.fabric.utilities.rank_zero import rank_zero_only
from datasets import Dataset, load_dataset
from typing import Dict, Any

from src.model import Pico
from .smlmt import SMLMTTask, ClassifierMLP
from src.training.utils import (
    initialize_run_dir,
    initialize_fabric,
    initialize_configuration,
    initialize_dataset,
    initialize_tokenizer,
    initialize_dataloader,
    initialize_lr_scheduler,
    initialize_hf_checkpointing,
    initialize_experiment_tracker,
    initialize_logging,
    initialize_optimizer,
)
from src.checkpointing import (
    load_checkpoint,
    save_checkpoint,
    save_evaluation_results,
    compute_learning_dynamics_states,
    save_learning_dynamics_states,
)

from src.evaluation import run_evaluation

import higher


class Trainer:
    def __init__(self, config_path: str):
        """
        Initializes the Trainer class. This Trainer class implements a `train` method, which is the
        main entry point for training the Pico model. Before calling `train`, the Trainer class
        initializes the following:

            - Configuration loading and validation
            - Model, optimizer, and dataset setup
            - Logging and experiment tracking setup
            - Checkpoint management

        Args:
            config_path (str): Path to the YAML configuration file containing any overrides.
        """

        ########################################################
        #
        # Basic Initialization of Configs, Fabric, Model, Optimizer, etc.
        #
        ########################################################

        # Setup Config
        self.configs = initialize_configuration(config_path)

        # Reset the RoPE frequency tensor cache so it uses the current max_seq_len.
        from src.model.pico import RoPE

        RoPE._freqs_cis = None

        # Setup Run Directory (i.e. where we store checkpoints, logs, etc.)
        initialize_run_dir(checkpointing_config=self.configs["checkpointing"])

        # Setup Logger
        self.experiment_tracker = initialize_experiment_tracker(
            monitoring_config=self.configs["monitoring"],
            checkpointing_config=self.configs["checkpointing"],
        )

        # Setup Fabric
        self.fabric = initialize_fabric(
            training_config=self.configs["training"],
            experiment_tracker=self.experiment_tracker,
        )
        L.seed_everything(42, verbose=False)

        # Set up logging
        self.logger = initialize_logging(
            monitoring_config=self.configs["monitoring"],
            checkpointing_config=self.configs["checkpointing"],
            fabric=self.fabric,
        )

        # Setup Model, Optimizer, and Dataloaders
        self.model = Pico(model_config=self.configs["model"], fabric=self.fabric)
        self.optimizer = initialize_optimizer(
            training_config=self.configs["training"], model=self.model
        )
        self.lr_scheduler = initialize_lr_scheduler(
            training_config=self.configs["training"], optimizer=self.optimizer
        )

        # 3. If SMLMT is enabled, instantiate the classifier (so model.state_dict has its keys).
        self.smlmt_enabled = self.configs["smlmt"].enabled
        if self.smlmt_enabled:
            self.smlmt_probability = float(self.configs["smlmt"].probability)
            self.smlmt_num_classes = int(self.configs["smlmt"].num_classes)
            self.smlmt_support = int(self.configs["smlmt"].support_per_class)
            self.smlmt_query = int(self.configs["smlmt"].query_per_class)
            # Read hidden dimensions and dropout from config, with defaults if not provided.
            self.hidden_dims = self.configs["smlmt"].hidden_dims
            self.dropout = self.configs["smlmt"].dropout
            self.model.classifier_smlmt = ClassifierMLP(
                input_dim=self.configs["model"].d_model,
                hidden_dims=self.hidden_dims,
                num_classes=self.smlmt_num_classes,
                dropout=self.dropout,
            )
            if self.fabric._precision == "bf16-mixed":  # check BF16
                dtype = torch.bfloat16
                device = self.fabric.device
                self.model.classifier_smlmt = self.model.classifier_smlmt.to(
                    device, dtype=dtype
                )
            print(f"SMLMT enabled with probability {self.smlmt_probability}")

        # Wrap with Fabric
        self.model, self.optimizer = self.fabric.setup(self.model, self.optimizer)

        ########################################################
        #
        # Boilerplate to deal with loading/resuming from checkpoints
        #
        ########################################################

        self.should_load_checkpoint = self.configs["checkpointing"].training.auto_resume

        # Possibly load a checkpoint
        if self.should_load_checkpoint:
            resume_checkpoint = load_checkpoint(
                checkpointing_config=self.configs["checkpointing"],
                checkpoint_step="latest",
                fabric=self.fabric,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
            )

            if resume_checkpoint:
                (
                    self.model,
                    self.optimizer,
                    self.lr_scheduler,
                    self.initial_batch_step,
                ) = resume_checkpoint
            else:
                self.initial_batch_step = 0
        else:
            self.initial_batch_step = 0

        ########################################################
        #
        # Initialization of Dataset & DataLoader (possibly fast-forwarding to correct batch)
        #
        ########################################################

        self.train_dataset, fast_forward_steps = initialize_dataset(
            data_config=self.configs["data"],
            fabric=self.fabric,
            initial_batch_step=self.initial_batch_step,
            return_fast_forward_steps=True,
        )
        self.train_dataloader = initialize_dataloader(
            data_config=self.configs["data"],
            training_config=self.configs["training"],
            fabric=self.fabric,
            dataset=self.train_dataset,
        )
        self.train_dataloader = self.fabric.setup_dataloaders(
            self.train_dataloader, use_distributed_sampler=False
        )
        self.tokenizer = initialize_tokenizer(data_config=self.configs["data"])

        if self.smlmt_enabled:
            # If sentences are provided in the config, use them; otherwise, extract from train_dataset.
            if self.configs["smlmt"].sentences:
                self.smlmt_sentences = self.configs["smlmt"]["sentences"]
            else:
                self.smlmt_sentences = []
                max_samples = 10000
                max_seq_len = self.configs["model"].max_seq_len  # e.g., 1024
                # Try using indexing if the dataset supports __len__ and __getitem__
                try:
                    dataset_length = len(self.train_dataset)
                    collected = 0
                    for i in range(dataset_length):
                        example = self.train_dataset[i]
                        if "text" in example:
                            tokenized = self.tokenizer.tokenize(example)
                            if len(tokenized) <= max_seq_len:
                                self.smlmt_sentences.append(example)
                                collected += 1
                        elif "input_ids" in example:
                            decoded = self.tokenizer.decode(example["input_ids"])
                            tokenized = self.tokenizer.tokenize(decoded)
                            if len(tokenized) <= max_seq_len:
                                self.smlmt_sentences.append(decoded)
                                collected += 1
                        # Stop once we've collected enough valid sequences.
                        if collected >= max_samples:
                            break
                except (TypeError, AttributeError):
                    # Fallback for iterable datasets that do not support len() or indexing.
                    samples_collected = 0
                    iterator = iter(self.train_dataset)
                    while samples_collected < max_samples:
                        try:
                            example = next(iterator)
                        except StopIteration:
                            break
                        if "text" in example:
                            tokenized = self.tokenizer.tokenize(example)
                            if len(tokenized) <= max_seq_len:
                                self.smlmt_sentences.append(example)
                                samples_collected += 1
                        elif "input_ids" in example:
                            decoded = self.tokenizer.decode(example["input_ids"])
                            tokenized = self.tokenizer.tokenize(decoded)
                            if len(tokenized) <= max_seq_len:
                                self.smlmt_sentences.append(decoded)
                                samples_collected += 1

            if not self.configs["smlmt"].vocabulary:
                # For example, sample 1000 words from the tokenizer's vocabulary.
                full_vocab = list(self.tokenizer.get_vocab().keys())
                self.smlmt_vocabulary = random.sample(
                    full_vocab, min(1000, len(full_vocab))
                )
            else:
                self.smlmt_vocabulary = self.configs["smlmt"].vocabulary

        # Setup HuggingFace Checkpointing
        if self.configs["checkpointing"].save_checkpoint_repo_id is not None:
            initialize_hf_checkpointing(
                checkpointing_config=self.configs["checkpointing"], fabric=self.fabric
            )

        # NOTE: We may need to fast-forward the iterator to the correct step so that we can
        # continue from the correct batch of data we would have seen had training not
        # previously stopped.
        train_iterator = iter(self.train_dataloader)
        if fast_forward_steps > 0:
            fast_forward_sub_steps = (
                fast_forward_steps
                * self.configs["training"].optimization.gradient_accumulation_steps
            )
            for _ in range(fast_forward_sub_steps):
                next(train_iterator)

        self.train_iterator = train_iterator

        # NOTE: Sychronizing processes after fast-forwarding iterator
        self.fabric.barrier()

        ########################################################
        #
        # Helper flags used during training for checkpointing and evaluation
        #
        ########################################################

        # Helper flag to determine if we should evaluate the model
        self.should_evaluate = (
            self.configs["evaluation"].metrics is not None
            and len(self.configs["evaluation"].metrics) > 0
        )

        self.should_compute_learning_dynamics = (
            self.configs["checkpointing"].learning_dynamics.layer_suffixes is not None
            and len(self.configs["checkpointing"].learning_dynamics.layer_suffixes) > 0
        )

        if self.should_compute_learning_dynamics:
            if self.configs["checkpointing"].learning_dynamics.eval_data is not None:
                self.learning_dynamics_eval_dataset = load_dataset(
                    self.configs["checkpointing"].learning_dynamics.eval_data,
                    split="val",
                )
            else:
                self.learning_dynamics_eval_dataset = None

    def train(self) -> None:
        """Execute the main training workflow.

        This method orchestrates the complete training process by:
        1. Creating an initial checkpoint to save the starting state and evaluate the model as a
            baseline
        2. Running the main training loop via `_training_loop`
        3. Handling final checkpointing and evaluation

        The training progress is tracked through checkpoints and evaluations
        at intervals specified in the configuration.
        """

        ########################################################
        #
        # Initial Checkpointing and Evaluation
        #
        ########################################################

        # Save Initial Checkpoint -- If the checkpoint already exists, this performs a no-op
        save_checkpoint(
            configs=self.configs,
            checkpoint_step=self.initial_batch_step,
            fabric=self.fabric,
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            tokenizer=self.tokenizer,
            upload_logs=False,
        )

        # Save Initial Evaluation Results
        if self.should_evaluate:
            if self.initial_batch_step == 0:
                evaluation_results = run_evaluation(
                    evaluation_config=self.configs["evaluation"],
                    checkpointing_config=self.configs["checkpointing"],
                    fabric=self.fabric,
                    model=self.model,
                )
                self._log_evaluation_results(
                    evaluation_results, self.initial_batch_step
                )
                save_evaluation_results(
                    checkpointing_config=self.configs["checkpointing"],
                    fabric=self.fabric,
                    evaluation_results=evaluation_results,
                    checkpoint_step=self.initial_batch_step,
                )
            else:
                # NOTE: If the run crashed while evaluating, we need to restart the evaluation
                eval_results_path = os.path.join(
                    self.configs["checkpointing"].evaluation.eval_results_dir,
                    f"step_{self.initial_batch_step}.json",
                )
                if not os.path.exists(eval_results_path):
                    evaluation_results = run_evaluation(
                        evaluation_config=self.configs["evaluation"],
                        checkpointing_config=self.configs["checkpointing"],
                        fabric=self.fabric,
                        model=self.model,
                    )
                    self._log_evaluation_results(
                        evaluation_results, self.initial_batch_step
                    )
                    save_evaluation_results(
                        checkpointing_config=self.configs["checkpointing"],
                        fabric=self.fabric,
                        evaluation_results=evaluation_results,
                        checkpoint_step=self.initial_batch_step,
                    )

        ########################################################
        #
        # Main Training Loop (see `_training_loop` for details)
        #
        ########################################################

        if self.initial_batch_step < self.configs["training"].max_steps:
            self._log_training_configuration()
            final_step = self._training_loop()
        else:
            final_step = self.initial_batch_step

        ########################################################
        #
        # Final Checkpointing and Evaluation
        #
        ########################################################

        # Save Learning Dynamics States
        if self.should_compute_learning_dynamics:
            if self.learning_dynamics_eval_dataset is not None:
                self.log(f"Step {final_step} -- 📈 Saving Learning Dynamics")
                learning_dynamics_val_states = compute_learning_dynamics_states(
                    checkpointing_config=self.configs["checkpointing"],
                    fabric=self.fabric,
                    model=self.model,
                    dataset=self.learning_dynamics_eval_dataset,
                    compute_gradients=False,
                )
                save_learning_dynamics_states(
                    checkpointing_config=self.configs["checkpointing"],
                    fabric=self.fabric,
                    learning_dynamics_states=learning_dynamics_val_states,
                    checkpoint_step=final_step,
                    prefix="val",
                )

        # Handle checkpointing and final evaluation
        if final_step % self.configs["checkpointing"].save_every_n_steps != 0:
            self.log(f"Step {final_step} -- 💾 Saving Final Checkpoint")
            save_checkpoint(
                configs=self.configs,
                checkpoint_step=final_step,
                fabric=self.fabric,
                model=self.model,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                tokenizer=self.tokenizer,
            )

            # Final evaluation
            if self.should_evaluate:
                evaluation_results = run_evaluation(
                    evaluation_config=self.configs["evaluation"],
                    checkpointing_config=self.configs["checkpointing"],
                    fabric=self.fabric,
                    model=self.model,
                )
                self._log_evaluation_results(evaluation_results, final_step)
                save_evaluation_results(
                    checkpointing_config=self.configs["checkpointing"],
                    checkpoint_step=final_step,
                    fabric=self.fabric,
                    evaluation_results=evaluation_results,
                )

        self.log(f"🎉 Training complete! Final step: {final_step}")

        if final_step < self.configs["training"].max_steps:
            self.log(
                f"\t Note: Training stopped before max steps ({self.configs['training'].max_steps})",
                level=logging.WARNING,
            )

        # Cleanup distributed training
        self.fabric.barrier()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if torch.distributed.is_initialized():
                torch.distributed.destroy_process_group()

            del self.train_dataloader  # NOTE: shutting down worker nodes

        self.fabric.barrier()

    def _meta_step(
        self,
        support_texts,
        query_texts,
        support_labels: torch.Tensor,
        query_labels: torch.Tensor,
        batch_step: int,
    ) -> torch.Tensor:
        rank = self.fabric.global_rank
        world_size = self.fabric.world_size

        local_support_texts = support_texts[rank::world_size]
        local_query_texts = query_texts[rank::world_size]
        local_support_labels = support_labels[rank::world_size]
        local_query_labels = query_labels[rank::world_size]

        support_inputs = self.tokenizer(
            local_support_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.configs["smlmt"].max_length,
        )
        query_inputs = self.tokenizer(
            local_query_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.configs["smlmt"].max_length,
        )
        for key in support_inputs:
            support_inputs[key] = support_inputs[key].to(self.fabric.device)
        for key in query_inputs:
            query_inputs[key] = query_inputs[key].to(self.fabric.device)

        inner_lr = float(self.configs["smlmt"].inner_lr)
        inner_steps = int(self.configs["smlmt"].inner_steps)

        classifier_optimizer = torch.optim.Adam(
            self.model.classifier_smlmt.parameters(), lr=inner_lr
        )

        with higher.innerloop_ctx(
            self.model.classifier_smlmt, classifier_optimizer, track_higher_grads=True
        ) as (fclassifier, diffopt):
            # Define a helper function for the support forward pass.
            def support_forward(input_ids, attention_mask):
                return self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    return_hidden=True,
                )

            inner_losses = []
            inner_accuracies = []
            # --- Inner loop: update classifier parameters on the support set ---
            for inner_step in range(inner_steps):
                # Use activation checkpointing to reduce memory usage.
                support_out = torch.utils.checkpoint.checkpoint(
                    support_forward,
                    support_inputs["input_ids"],
                    support_inputs["attention_mask"],
                    use_reentrant=False,
                )
                # Unpack output: assuming the model returns (output, hidden, extra)
                _, support_hidden, _ = support_out
                support_repr = support_hidden.mean(dim=1).bfloat16()
                support_preds = fclassifier(support_repr)
                support_loss = F.cross_entropy(support_preds, local_support_labels)
                support_pred_labels = support_preds.argmax(dim=1)
                support_accuracy = (
                    (support_pred_labels == local_support_labels).float().mean()
                )
                self.fabric.log(
                    "train/maml_inner_loss",
                    support_loss.item(),
                    step=(batch_step * inner_steps + inner_step),
                )
                self.fabric.log(
                    "train/maml_inner_accuracy",
                    support_accuracy.item(),
                    step=(batch_step * inner_steps + inner_step),
                )
                inner_losses.append(support_loss.item())
                inner_accuracies.append(support_accuracy.item())
                # Update the classifier parameters in the inner loop.
                diffopt.step(support_loss)
            avg_inner_accuracy = sum(inner_accuracies) / len(inner_accuracies)
            self.fabric.log(
                "train/maml_inner_accuracy_avg", avg_inner_accuracy, step=batch_step
            )

            # --- Query pass: compute meta loss using the updated classifier ---
            def query_forward(input_ids, attention_mask):
                return self.model(
                    input_ids,
                    attention_mask=attention_mask,
                    return_hidden=True,
                )

            query_out = torch.utils.checkpoint.checkpoint(
                query_forward,
                query_inputs["input_ids"],
                query_inputs["attention_mask"],
                use_reentrant=False,
            )
            _, query_hidden, _ = query_out
            query_repr = query_hidden.mean(dim=1).bfloat16()
            query_preds = fclassifier(query_repr)
            meta_loss = F.cross_entropy(query_preds, local_query_labels)

        # Aggregate the meta loss across GPUs.
        meta_loss_agg = self.fabric.all_reduce(meta_loss, reduce_op="mean")
        return meta_loss_agg

    def _training_loop(self) -> int:
        """Execute the main training loop.

        This method orchestrates the core training loop and includes the following features:
            - Gradient accumulation
            - Gradient clipping
            - Periodic model evaluation and checkpointing
            - Learning Dynamics Checkpointing
            - Learning rate scheduling
            - Logging of training metrics including loss and learning rate
            - Handling of infinite/NaN losses

        Returns:
            int: The final step count reached during training.
                NOTE: A complete training run should match the configured max_steps.
        """
        # Setup training loop variables
        batch_step = self.initial_batch_step

        # Interval accumulators for logging
        interval_loss = torch.tensor(0.0, device=self.fabric.device)  # supervised loss
        interval_steps = torch.tensor(0, device=self.fabric.device)
        interval_inf_or_nan_count = torch.tensor(0, device=self.fabric.device)
        interval_smlmt_loss = torch.tensor(0.0, device=self.fabric.device)  # meta loss
        interval_smlmt_steps = torch.tensor(0, device=self.fabric.device)

        if self.should_compute_learning_dynamics:
            training_batch = {"input_ids": []}

        # Determine starting sub-batch step from gradient accumulation.
        initial_sub_batch_step = (
            batch_step
            * self.configs["training"].optimization.gradient_accumulation_steps
        )

        ###############################################################
        # Main Training Loop
        ###############################################################
        for sub_batch_step, sub_batch in enumerate(
            self.train_iterator, start=initial_sub_batch_step
        ):
            # --- 1. Supervised Branch (always computed) ---
            _input_ids = sub_batch["input_ids"].clone().detach().to(self.fabric.device)
            input_ids = _input_ids[:, :-1]
            labels = _input_ids[:, 1:]

            # Optionally, store training batch for learning dynamics.
            if self.should_compute_learning_dynamics:
                gathered_input_ids = self.fabric.all_gather(_input_ids)
                if self.fabric.world_size > 1:
                    gathered_input_ids = gathered_input_ids.reshape(
                        -1, *gathered_input_ids.shape[2:]
                    )
                training_batch["input_ids"].extend(gathered_input_ids.tolist())

            # Forward pass for supervised loss.
            model_output, _ = self.model(input_ids)
            model_output = model_output.transpose(1, 2)
            supervised_loss = F.cross_entropy(model_output, labels)

            # --- 2. Optional MAML (SMLMT) Meta–Loss ---
            meta_loss = None
            if self.smlmt_enabled:
                # Only rank 0 decides whether to trigger the meta branch.
                if self.fabric.global_rank == 0:
                    flag_value = (
                        1.0 if random.random() < self.smlmt_probability else 0.0
                    )
                else:
                    flag_value = 0.0  # Placeholder for non-rank0 processes.
                flag_tensor = torch.tensor(flag_value, device=self.fabric.device)
                flag_tensor = self.fabric.broadcast(flag_tensor, src=0)
                should_compute_meta = bool(flag_tensor.item() > 0.5)
                if should_compute_meta:
                    self.log("MAML SMLMT branch triggered", level=logging.INFO)

                    # Generate one SMLMT task (episode)
                    task_generator = SMLMTTask(
                        self.smlmt_sentences,
                        self.smlmt_vocabulary,
                        num_classes=self.smlmt_num_classes,
                        support_per_class=self.smlmt_support,
                        query_per_class=self.smlmt_query,
                        mask_token=self.tokenizer.mask_token
                        if hasattr(self.tokenizer, "mask_token")
                        else "[MASK]",
                    )
                    support_set, query_set = task_generator.generate_task()
                    support_texts = [sent for (sent, _) in support_set]
                    query_texts = [sent for (sent, _) in query_set]
                    support_labels = torch.tensor(
                        [label for (_, label) in support_set], device=self.fabric.device
                    )
                    query_labels = torch.tensor(
                        [label for (_, label) in query_set], device=self.fabric.device
                    )
                    try:
                        meta_loss = self._meta_step(
                            support_texts,
                            query_texts,
                            support_labels,
                            query_labels,
                            batch_step,
                        )
                    except Exception as e:
                        self.log(f"Error in meta–step: {e}", level=logging.ERROR)
                        meta_loss = torch.tensor(0.0, device=self.fabric.device)

                    interval_smlmt_loss += meta_loss.item()
                    interval_smlmt_steps += 1
                    self.fabric.log(
                        "train/smlmt_loss", meta_loss.item(), step=batch_step
                    )

            # --- 3. Gradient Accumulation: Combine Losses and Backward ---
            # Here, we add the losses (if meta_loss is computed) so that gradients flow from both.
            total_loss = supervised_loss
            if meta_loss is not None:
                total_loss = total_loss + meta_loss

            should_accumulate_gradients = (sub_batch_step + 1) % self.configs[
                "training"
            ].optimization.gradient_accumulation_steps != 0

            self.fabric.backward(
                total_loss
                / self.configs["training"].optimization.gradient_accumulation_steps,
                model=self.model,
            )
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                interval_inf_or_nan_count += 1
            else:
                interval_loss += supervised_loss.item()  # accumulate supervised loss
                interval_steps += 1

            # If we're still accumulating gradients, skip the optimizer step.
            if should_accumulate_gradients:
                continue

            # --- 4. Logging ---
            if batch_step % self.configs["monitoring"].logging.log_every_n_steps == 0:
                self._log_training_metrics(
                    interval_loss=interval_loss,
                    interval_steps=interval_steps,
                    interval_inf_or_nan_count=interval_inf_or_nan_count,
                    interval_smlmt_loss=interval_smlmt_loss,
                    interval_smlmt_steps=interval_smlmt_steps,
                    batch_step=batch_step,
                )
                # Reset interval accumulators.
                interval_smlmt_loss = torch.tensor(0.0, device=self.fabric.device)
                interval_smlmt_steps = torch.tensor(0, device=self.fabric.device)
                interval_loss = torch.tensor(0.0, device=self.fabric.device)
                interval_steps = torch.tensor(0, device=self.fabric.device)
                interval_inf_or_nan_count = torch.tensor(0, device=self.fabric.device)

            # --- 5. Learning Dynamics Checkpointing ---
            if batch_step % self.configs["checkpointing"].save_every_n_steps == 0:
                if self.should_compute_learning_dynamics:
                    self.log(f"Step {batch_step} -- 📈 Saving Learning Dynamics")
                    training_batch_dataset = Dataset.from_dict(training_batch)
                    learning_dynamics_train_states = compute_learning_dynamics_states(
                        checkpointing_config=self.configs["checkpointing"],
                        fabric=self.fabric,
                        model=self.model,
                        dataset=training_batch_dataset,
                        compute_gradients=True,
                    )
                    save_learning_dynamics_states(
                        checkpointing_config=self.configs["checkpointing"],
                        checkpoint_step=batch_step,
                        prefix="train",
                        fabric=self.fabric,
                        learning_dynamics_states=learning_dynamics_train_states,
                        learning_dynamics_dataset=training_batch_dataset,
                        tokenizer=self.tokenizer,
                    )
                    training_batch = {"input_ids": []}  # Reset for next batch

                    # Also save validation learning dynamics if available.
                    if self.learning_dynamics_eval_dataset is not None:
                        learning_dynamics_val_states = compute_learning_dynamics_states(
                            checkpointing_config=self.configs["checkpointing"],
                            fabric=self.fabric,
                            model=self.model,
                            dataset=self.learning_dynamics_eval_dataset,
                            compute_gradients=False,
                        )
                        save_learning_dynamics_states(
                            checkpointing_config=self.configs["checkpointing"],
                            checkpoint_step=batch_step,
                            prefix="val",
                            fabric=self.fabric,
                            learning_dynamics_states=learning_dynamics_val_states,
                        )

            # --- 6. Optimization Step ---
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

            batch_step += 1

            # --- 7. Checkpointing and Evaluation ---
            if batch_step % self.configs["checkpointing"].save_every_n_steps == 0:
                self.log(f"Step {batch_step} -- 💾 Saving Checkpoint")
                save_checkpoint(
                    configs=self.configs,
                    checkpoint_step=batch_step,
                    fabric=self.fabric,
                    model=self.model,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    tokenizer=self.tokenizer,
                )
                if self.should_evaluate:
                    evaluation_results = run_evaluation(
                        evaluation_config=self.configs["evaluation"],
                        checkpointing_config=self.configs["checkpointing"],
                        fabric=self.fabric,
                        model=self.model,
                    )
                    if evaluation_results is not None:
                        self._log_evaluation_results(evaluation_results, batch_step)
                        save_evaluation_results(
                            checkpointing_config=self.configs["checkpointing"],
                            fabric=self.fabric,
                            evaluation_results=evaluation_results,
                            checkpoint_step=batch_step,
                        )

            # Break out of the loop if we've reached the maximum training steps.
            if batch_step >= self.configs["training"].max_steps:
                break

        return batch_step

    ########################################################
    #
    # Trainer Logging Functinalities
    #
    ########################################################

    def _log_training_metrics(
        self,
        interval_loss: torch.Tensor,
        interval_steps: torch.Tensor,
        interval_inf_or_nan_count: torch.Tensor,
        interval_smlmt_loss: torch.Tensor,  # NEW: accumulated SMLMT loss over the interval
        interval_smlmt_steps: torch.Tensor,  # NEW: number of SMLMT updates in the interval
        batch_step: int,
    ):
        """
        Gathers together the training metrics computed across all processes in distributed training
        and logs them in a tree-style format.
        """
        gathered_interval_loss = self.fabric.all_reduce(
            interval_loss, reduce_op="sum"
        ).item()
        gathered_interval_inf_or_nan_count = self.fabric.all_reduce(
            interval_inf_or_nan_count, reduce_op="sum"
        ).item()
        gathered_interval_steps = self.fabric.all_reduce(
            interval_steps, reduce_op="sum"
        ).item()

        avg_supervised_loss = (
            gathered_interval_loss / gathered_interval_steps
            if gathered_interval_steps > 0
            else float("inf")
        )

        # Aggregate SMLMT loss metrics.
        gathered_interval_smlmt_loss = self.fabric.all_reduce(
            interval_smlmt_loss, reduce_op="sum"
        ).item()
        gathered_interval_smlmt_steps = self.fabric.all_reduce(
            interval_smlmt_steps, reduce_op="sum"
        ).item()
        avg_smlmt_loss = (
            gathered_interval_smlmt_loss / gathered_interval_smlmt_steps
            if gathered_interval_smlmt_steps > 0
            else float("inf")
        )

        self.fabric.log("train/supervised_loss", avg_supervised_loss, step=batch_step)
        self.fabric.log("train/smlmt_loss", avg_smlmt_loss, step=batch_step)
        self.fabric.log(
            "trainer/inf_or_nan_count",
            gathered_interval_inf_or_nan_count,
            step=batch_step,
        )
        self.fabric.log(
            "trainer/learning_rate",
            self.lr_scheduler.get_last_lr()[0],
            step=batch_step,
        )

        # Log to console in tree format.
        self.log(f"Step {batch_step} -- 🔄 Training Metrics")
        self.log(f"├── Supervised Loss: {avg_supervised_loss:.4f}")
        self.log(f"├── SMLMT Loss: {avg_smlmt_loss:.4f}")
        self.log(f"├── Learning Rate: {self.lr_scheduler.get_last_lr()[0]:.2e}")
        self.log(f"└── Inf/NaN count: {gathered_interval_inf_or_nan_count}")

    def _log_evaluation_results(
        self, evaluation_results: Dict[str, Any], batch_step: int
    ):
        """Log model evaluation metrics to experiment tracking system and console."""
        if self.fabric.global_rank == 0:
            self.log(f"Step {batch_step} -- 📊 Evaluation Results")
            for i, (metric, result) in enumerate(evaluation_results.items()):
                prefix = "└──" if i == len(evaluation_results) - 1 else "├──"
                if metric == "universal_ner":
                    overall_f1 = result.get("detailed", {}).get("overall_f1", None)
                    overall_acc = result.get("detailed", {}).get(
                        "overall_accuracy", None
                    )
                    simple_report = {"f1": overall_f1, "overall_accuracy": overall_acc}
                    self.log(f"{prefix} {metric}: {simple_report}")
                    self.fabric.log(f"eval/{metric}", simple_report, step=batch_step)
                else:
                    self.log(f"{prefix} {metric}: {result}")
                    self.fabric.log(f"eval/{metric}", result, step=batch_step)

    def _log_training_configuration(self):
        """Log training configuration details including model, hardware, and batch settings."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        global_batch_size = self.configs["data"].dataloader.batch_size
        per_device_batch_size = self.train_dataloader.batch_size
        gradient_accumulation_steps = self.configs[
            "training"
        ].optimization.gradient_accumulation_steps

        device_type = ""
        fabric_device = str(self.fabric.device)
        if torch.cuda.is_available() and "cuda" in fabric_device:
            device_type = torch.cuda.get_device_name(self.fabric.device)
        elif torch.backends.mps.is_available() and "mps" in fabric_device:
            device_type = "MPS (Apple Silicon)"
        else:
            device_type = "CPU"

        self.log("=" * 50)
        self.log("✨ Training Configuration")
        self.log("=" * 50)
        self.log(f"Starting from step: {self.initial_batch_step}")
        self.log("Model Setup:")
        self.log(f"└─ Total Parameters: {total_params:,}")
        self.log(f"└─ Trainable Parameters: {trainable_params:,}")
        self.log("Distributed Setup:")
        self.log(f"└─ Number of Devices: {self.fabric.world_size}")
        self.log(f"└─ Device Type: {device_type}")
        self.log("Batch Size Configuration:")
        self.log(f"└─ Global Batch Size: {global_batch_size}")
        self.log(f"└─ Per Device Batch Size: {per_device_batch_size}")
        self.log(f"└─ Gradient Accumulation Steps: {gradient_accumulation_steps}")
        self.log("=" * 50)

    @rank_zero_only
    def log(self, msg: str, level: int = logging.INFO) -> None:
        """Log messages only from rank zero process."""
        self.logger.log(level, msg)
