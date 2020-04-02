import matplotlib.pyplot as plt
import os

def make_graphs(models, test_number, testfile, xlabel, ylabel):
    for i,model in enumerate(models):
        test_file = 'plot_' + test + '_data.txt'
        file = os.path.join(os.getcwd(), 'model' + str(model) + '_t'+ str(test_number), test_file)
        with open(file, 'r') as f:
            data = []
            # print(f.readlines())
            for row in f.readlines():
                # print(row)
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
    fig = plt.gcf()
    # fig.savefig(os.path.join(self._path, 'plot_'+filename+'.png'), dpi=self._dpi)
    filename = testfile + '_t' + str(test_number)
    fig.savefig('plot_'+filename+'.png', dpi=96)
    plt.close("all")
    print("Saved figure : {}".format(filename))

if __name__ == "__main__":

    models = [3, 6, 8]
    test_n = [1, 2, 3]
    tests = ["queue", "reward", "waitingTime"]
    xlabel = {"queue": 'Step', "reward": "Action step", "waitingTime": "Action step"}
    ylabel = {"queue": 'Queue Lenght (vehicules)', "reward": "Reward", "waitingTime": "Cumulative waiting time (s)"}
    
    for test in tests:
        for n in test_n:
            make_graphs(models, n, test, xlabel[test], ylabel[test])
