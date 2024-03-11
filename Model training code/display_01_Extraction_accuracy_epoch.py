import json
import csv
import os


def read_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def process_data(data, result_data, filename):
    unique_epochs = set()

    log_history = data.get('log_history', [])

    for entry in log_history:
        if isinstance(entry, dict):
            epoch = entry.get('epoch')
            accuracy = entry.get('eval_accuracy')
            if epoch is not None and accuracy is not None and epoch not in unique_epochs:
                unique_epochs.add(epoch)
                result_data.setdefault(epoch, {}).update({filename: accuracy})


def save_to_csv(output_csv_file, data):
    with open(output_csv_file, 'w', newline='') as csv_file:
        fieldnames = ['epoch'] + list(set(file for values in data.values() for file in values))
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        #Write data by epoch to CSV file
        for epoch, values in data.items():
            row = {'epoch': epoch}
            row.update(values)
            writer.writerow(row)


def process_multiple_json_files(json_folder, output_csv_file):
    result_data = {}

    # Iterate over all JSON files in the folder
    for filename in os.listdir(json_folder):
        if filename.endswith(".json"):
            filename_without_extension = os.path.splitext(filename)[0]   # Remove the json suffix
            json_file_path = os.path.join(json_folder, filename)
            json_data = read_json(json_file_path)
            process_data(json_data, result_data, filename_without_extension)

    # Save results to CSV file
    save_to_csv(output_csv_file, result_data)


if __name__ == "__main__":
    json_folder = 'content/data/daic_rmse/polling4'  # Replace with your folder path
    output_csv_file = 'csv/polling4.csv'
    process_multiple_json_files(json_folder, output_csv_file)
