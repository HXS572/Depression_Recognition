"""
training
"""

import argparse
import yaml
import os
import shutil

import preprocess_data
import build_model
import evaluate

from transformers import (
    Trainer,
    is_apex_available,
    TrainingArguments,
    set_seed,
)
from datasets import load_dataset

from typing import Any, Dict, Union

import torch
from packaging import version
from torch import nn


# Get the configuration file
parser = argparse.ArgumentParser()
parser.add_argument('--config', help='yaml configuration file path') # EXAMPLE content/config/test.yaml
args = parser.parse_args()
config_file = args.config

with open(config_file) as f:
    configuration = yaml.load(f, Loader=yaml.FullLoader)
print('Loaded configuration file: ', config_file)
print(configuration)


set_seed(configuration['seed'])

# Prepare data
train_filepath = configuration['output_dir'] + "/splits/train.csv"
test_filepath = configuration['output_dir'] + "/splits/test.csv"
valid_filepath = configuration['output_dir'] + "/splits/valid.csv"

if not os.path.exists(train_filepath) or not os.path.exists(test_filepath) or not os.path.exists(valid_filepath):
    import prepare_data
    # prepare datasplits
    df = prepare_data.df(configuration['corpora'], configuration['data_path'])
    prepare_data.prepare_splits(df, configuration)


# Save used configuraton inside the model folder
shutil.copy(config_file, configuration['output_dir'])


# Prepare data splits for Training
train_dataset, eval_dataset, input_column, output_column, label_list, num_labels = preprocess_data.training_data(configuration)
# Load the processor of the underlying pretrained wav2vec model
config, processor, target_sampling_rate = preprocess_data.load_processor(configuration, label_list, num_labels)
# Get the preprocessed data splits
train_dataset, eval_dataset = preprocess_data.preprocess_data(configuration, processor, target_sampling_rate, train_dataset, eval_dataset, input_column, output_column, label_list)


# Define the data collator
data_collator = build_model.data_collator(processor)

# Define evaluation metrics
compute_metrics = build_model.compute_metrics

# Load the pretrained checkpoint
model = build_model.load_pretrained_checkpoint(config, configuration['processor_name_or_path'])

if configuration['freeze_feature_extractor']:
    model.freeze_feature_extractor()
    print("freeze feature extractor")



training_args = TrainingArguments(
    output_dir=configuration['output_dir'],
    per_device_train_batch_size=configuration['per_device_train_batch_size'],
    per_device_eval_batch_size=configuration['per_device_eval_batch_size'],
    gradient_accumulation_steps=configuration['gradient_accumulation_steps'],
    evaluation_strategy=configuration['evaluation_strategy'],
    num_train_epochs=configuration['num_train_epochs'],
    fp16=configuration['fp16'],
    save_steps=configuration['save_steps'],
    eval_steps=configuration['eval_steps'],
    logging_steps=configuration['logging_steps'],
    learning_rate=float(configuration['learning_rate']),
    save_total_limit=configuration['save_total_limit'],
    seed=configuration['seed'],
    data_seed=configuration['seed'],
    report_to=configuration.get('report_to', None),
    load_best_model_at_end=True
)


"""For future use we can create our training script, we do it in a simple way. You can add more on you own."""
if is_apex_available():
    from apex import amp

if version.parse(torch.__version__) >= version.parse("1.6"):
    _is_native_amp_available = True
    from torch.cuda.amp import autocast

class CTCTrainer(Trainer):
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:

        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.use_amp:
            with autocast():
                loss = self.compute_loss(model, inputs)
        else:
            loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        if self.use_amp:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            self.deepspeed.backward(loss)
        else:
            loss.backward()

        return loss.detach()        # .detach() 截断反向传播的梯度流


# Pass all instances to Trainer
trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)

# Train
try:
    trainer.train(resume_from_checkpoint=configuration['resume_from_checkpoint'])
except RuntimeError as exception:
    if "out of memory" in str(exception):
        print("WANRING: out of memory")
        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()
        else:
            raise exception


# Evaluate
# Load test data
test_dataset = load_dataset("csv",
    data_files={"test": test_filepath},
    delimiter="\t",
    cache_dir=configuration['cache_dir'])["test"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Load model configuration, processor, and pretrained checkpoint
config, processor, model = evaluate.load_model(configuration, device)

# Resample the audio files
test_dataset = test_dataset.map(evaluate.speech_file_to_array_fn,
    fn_kwargs=dict(processor=processor)
    )

# get predictions
result = test_dataset.map(evaluate.predict,
    batched=True,
    batch_size=8,
    fn_kwargs=dict(configuration=configuration, processor=processor, model=model, device=device)
    )

label_names = [config.id2label[i] for i in range(config.num_labels)]
print(label_names)

y_true = [config.label2id[name] for name in result["class_4"]]
y_pred = result["predicted"]

print("True values: \t", y_true[:5])
print("Predicted values: \t", y_pred[:5])

from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_true, y_pred, target_names=label_names))
evaluate.report(configuration, y_true, y_pred, label_names)
