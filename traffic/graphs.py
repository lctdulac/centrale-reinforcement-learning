import matplotlib.pyplot as plt
import os

def make_graphs(models, testnumner, testfile, xlabel, ylabel):
    for i,model in enumerate(models):
        test_file = 'plot_' + test + '_data.txt'
        file = os.path.join(os.getcwd(), 'models/model_' + str(model) + 't'+ str'test_n', test_file)
        with open(file, 'r') as f:
            data = []
            # print(f.readlines())
            for row in f.readlines():
            
                data.append(float(row))
            plt.plot(data)
            if i ==0:
                min_val = min(data)
                max_val = max(data)
            else:
                m = min(data)
                M = max(data)
                if m < min_val:
                    min_val = m
                if M > max_val:
                    max_val = M
    plt.legend(["model1", "model2", "model3"])
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.margins(0)
    plt.ylim(min_val - 0.05 * abs(min_val), max_val + 0.05 * abs(max_val))
    plt.gcf()
    # fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
    plt.show()

if __name__ == "__main__":

    models = [3, 6, 8]
    test = "queue"
    xlabel = 'Action step'
    ylabel = 'Cumulative waiting time (s)'

    make_graphs(models, test, xlabel, ylabel)