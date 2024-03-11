import pandas as pd
import matplotlib.pyplot as plt


def plot_csv_data(csv_file):
   
    df = pd.read_csv(csv_file)

    ## Extract epoch and all non-epoch columns
    epochs = df['epoch']
    accuracy_columns = [col for col in df.columns if col != 'epoch']

    plt.figure(figsize=(6, 4))

    for acc_col in accuracy_columns:
        plt.plot(epochs, df[acc_col], label=acc_col)

    plt.title('Epoch vs Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')  
    # plt.legend()
    plt.grid(True)

    # Setting the font and size of image text
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 60

    plt.savefig('./content/data/fig/polling4.svg', dpi=600)
    plt.show()


if __name__ == "__main__":
    csv_file = 'csv/polling4.csv'  # Replace with the path to your CSV file
    plot_csv_data(csv_file)


