import matplotlib.pyplot as plt

if __name__ == '__main__':
    dataset = ['mnist_d', 'mnist_f', 'cifar_10', 'cifar_100_f', 'cifar_100_c']
    accuracy_ann = [91.14, 70.69, 34.5, 10.94, 22.03]
    accuracy_cnn = [99.1, 92.45, 70.5, 43.19, 52.66]

    fig = plt.figure(1)
    plt.bar(dataset, accuracy_ann, color='blue',
            width=0.4)

    plt.xlabel("Datasets")
    plt.ylabel("Accuracy %")
    plt.title("ANN Accuracy Plot")
    plt.ylim([0, 100])
    plt.savefig("ANN_Accuracy_Plot.pdf")

    plt.figure(2)
    plt.bar(dataset, accuracy_cnn, color='blue',
            width=0.4)

    plt.xlabel("Datasets")
    plt.ylabel("Accuracy %")
    plt.title("CNN Accuracy Plot")
    plt.ylim([0, 100])
    plt.savefig("CNN_Accuracy_Plot.pdf")
