import argparse
import yaml
import os


import prepare_data
from datasets import load_dataset
import pandas as pd
import build_model
from nested_array_catcher import nested_array_catcher

import torch
import torch.nn as nn
import torchaudio
import numpy as np

from numpy import savetxt
import librosa
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoConfig, Wav2Vec2Processor, set_seed


# Obtaining data for model testing
def get_test_data(configuration):
    test_filepath = configuration['output_dir'] + "/splits/test.csv"

    if not os.path.exists(test_filepath):
        df = prepare_data.df(configuration['corpora'],
            configuration['data_path'])
        prepare_data.prepare_splits(df, configuration)

    # Load test data
    test_dataset = load_dataset("csv",
                    data_files={"test": test_filepath},
                    delimiter="\t",
                    cache_dir=configuration['cache_dir']
                    )["test"]

    return test_dataset

# Load Model Configuration
def load_model(configuration, device):
    # Load model configuration, processor, and pretrained checkpoint
    processor_name_or_path = configuration['processor_name_or_path']
    model_name_or_path = configuration['output_dir'] + configuration['checkpoint']
    print('Loading checkoint: ', model_name_or_path)

    config = AutoConfig.from_pretrained(model_name_or_path,
                cache_dir=configuration['cache_dir']
                )
    processor = Wav2Vec2Processor.from_pretrained(processor_name_or_path,
                    cache_dir=configuration['cache_dir']
                    )
    model = build_model.Wav2Vec2ForSpeechClassification.from_pretrained(
                model_name_or_path,
                cache_dir=configuration['cache_dir']
                ).to(device)

    return config, processor, model

# Resample the audio files
def speech_file_to_array_fn(batch, processor):
    # The daic-woz dataset audio sampling rate is 16khz
    speech_array, sampling_rate = torchaudio.load(batch["path"])
    speech_array = speech_array.squeeze().numpy()
    speech_array = librosa.resample(np.asarray(speech_array),
                    sampling_rate,
                    processor.feature_extractor.sampling_rate
                    )

    speech_array = nested_array_catcher(speech_array)

    batch["speech"] = speech_array
    return batch


# Extract features using the processor
def predict(batch, configuration, processor, model, device):
    features = processor(batch["speech"],
                    sampling_rate=processor.feature_extractor.sampling_rate,
                    return_tensors="pt",
                    padding=True
                    )

    input_values = features.input_values.to(device)

    if configuration['return_attention_mask'] is not False:
        attention_mask = features.attention_mask.to(device)
    else:
        attention_mask = None

    # Pass input values through the model to get predictions
    # 通过模型传递输入值以获得预测
    with torch.no_grad():
        logits = model(input_values, attention_mask=attention_mask).logits
    torch.cuda.empty_cache()  # Memory freeing

    pred_ids = torch.argmax(logits, dim=-1).detach().cpu().numpy()
    batch["predicted"] = pred_ids
    return batch


import pandas as pd
from sklearn.metrics import mean_squared_error, roc_curve, auc
from math import sqrt
import matplotlib.pyplot as plt


# Used to output experimental results on model evaluation
def report(configuration, y_true, y_pred, label_names, labels=None):
    clsf_report = classification_report(y_true,
                    y_pred,
                    labels=labels,
                    target_names=label_names,
                    zero_division=0,
                    output_dict=True
                    )

    clsf_report_df = pd.DataFrame(clsf_report).transpose()
    print(clsf_report_df)
    cm = confusion_matrix(y_true, y_pred,)
    print(cm)
    cm = pd.DataFrame(cm, index=label_names, columns=label_names)
    print(cm)

    # Calculate MSE and RMSE
    mse = mean_squared_error(y_true, y_pred)
    rmse = sqrt(mse)

    # Adding MSE and RMSE to the Report
    clsf_report_df['MSE'] = mse
    clsf_report_df['RMSE'] = rmse

    # ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)

    # File save path. If it does not exist it is automatically created
    results_path = configuration['output_dir'] + '/results'
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # Plotting the roc curve
    """
    We only use it for drawing binary categorized roc. 
    you can comment that part of the code if you don't need OR to implement it elsewhere now.
    Then add the code you want.
    """
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(results_path + '/roc_curve.png')
    plt.close()

    # Save classification report and confusion matrix
    if ((configuration['test_corpora'] is not None) and (__name__ == '__main__')):
        file_name = (configuration['output_dir'].split('/')[-1] + '-evaluated-on-' + configuration['test_corpora']
                + '_clsf_report.csv')

        cm_file_name = (configuration['output_dir'].split('/')[-1] + '-evaluated-on-' + configuration['test_corpora']
                + '_conf_matrix.csv')
    else:
        file_name = 'clsf_report.csv'
        cm_file_name = 'conf_matrix.csv'

    clsf_report_df.to_csv(results_path + '/' + file_name, sep='\t')
    cm.to_csv(results_path + '/' + cm_file_name, sep='\t')


if __name__ == '__main__':
    # First, get the configuration file
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='yaml configuration file path')
    args = parser.parse_args()
    config_file = args.config

    with open(config_file) as f:
        configuration = yaml.load(f, Loader=yaml.FullLoader)
    print('Loaded configuration file: ', config_file)

    # Select a device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # To ensure reproducibility, this paper uniformly sets seed to 103.
    set_seed(configuration['seed'])
    config, processor, model = load_model(configuration, device)

    # Load data and evaluate model performance
    test_dataset = get_test_data(configuration)
    test_dataset = test_dataset.map(speech_file_to_array_fn,
        fn_kwargs=dict(processor=processor)
        )

    result = test_dataset.map(predict,
        batched=True,
        batch_size=8,
        fn_kwargs=dict(configuration=configuration,
                    processor=processor,
                    model=model,
                    device=device
                    )
        )

    label_names = [config.id2label[i] for i in range(config.num_labels)]
    # print(label_names)
    labels = list(config.id2label.keys())
    # print(labels)

    # True values and predicted values
    y_true = [config.label2id[name] for name in result["class_4"]]
    y_pred = result["predicted"]

    print("True values: \t", y_true[:5])
    print("Predicted values: \t", y_pred[:5])

    print(classification_report(y_true, y_pred, labels=labels, target_names=label_names))
    report(configuration, y_true, y_pred, label_names, labels)
