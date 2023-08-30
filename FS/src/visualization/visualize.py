import matplotlib.pyplot as plt


class plotScore:
    """define"""
    def __init__(self, data, feature_impt, pred_type):
        self.data = data
        self.feature_impt = feature_impt
        self.pred_type = pred_type.lower()

    def score_plot(self):
        plt.plot(range(1, len(self.data) + 1), self.data)
        plt.xlabel("Number of Features")
        score_type = "Accuracy" if self.pred_type == "classification" else "MSE"
        plt.ylabel(f"{score_type} Score")
        plt.title(f"{score_type} Score over Number of Features")

        # Adjust the x-axis labels to show every 5th tick
        n = 5
        x_ticks = range(1, len(self.data) + 1, n)
        plt.xticks(x_ticks)

        feature_index = self.data.index(max(self.data)) + 1 if self.pred_type == 'classification' else self.data.index(min(self.data)) + 1
        plt.axvline(x=feature_index, color='red', linestyle='dotted')
        # Display the x-axis number next to the axvline
        plt.show()
        if self.pred_type == 'classification':
            print(f"\nTop {self.data.index(max(self.data))+1} Features (Ordered by Feature Values):\n\n{self.feature_impt[:self.data.index(max(self.data))+1]}\n")
        else:
            print(f"\nTop {self.data.index(min(self.data))+1} Features (Ordered by Feature Values):\n\n{self.feature_impt[:self.data.index(min(self.data))+1]}\n")

class plotCurve:
    """define"""
    def __init__(self, data, label, ds_title, pred_type):
        self.data = [data]
        self.label = [label]
        self.ds_title = ds_title
        self.pred_type = pred_type.lower()

    def plot_line(self):
        for i, data in enumerate(self.data):
            plt.plot(range(1, len(data) + 1), data, label=f"Line {i+1}")

        plt.xlabel("Number of Features")
        score_type = "Accuracy" if self.pred_type == "classification" else "MSE"
        plt.ylabel(f"{score_type} Score")
        plt.title(f"{score_type} Score over Number of Features [{self.ds_title}]")
        n = 5
        x_ticks = range(1, len(self.data[0]) + 1, n)
        plt.xticks(x_ticks)
        plt.legend(self.label, bbox_to_anchor=(1.5, 1), loc='upper right')
        plt.show()

    def add_line(self, data, label):
        if isinstance(data, (int, float)):
            data = [data]
        self.label.append(label)
        self.data.append(data)
        self.plot_line()
