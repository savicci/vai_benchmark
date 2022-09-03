from matplotlib import pyplot as plt
import csv

COLS_NUM = 3


def create_plot(filename, ylabel, title,  save_filename):
    rows = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            else:
                rows.extend(row)

    x = []
    y = []
    for i in range(0, len(rows), COLS_NUM):
        x.append(float(rows[i]))
        y.append(float(rows[i + 1]))

    # plot
    plt.figure()
    plt.xlabel('Parameters')
    plt.ylabel(ylabel)

    plt.plot(x, y, 'b')

    plt.title(title)

    plt.savefig('plots/{}.png'.format(save_filename))
    # plt.show()


if __name__ == '__main__':
    create_plot('min_data.csv', 'Time [ms]', 'Minimal inference time', 'min_data')
    create_plot('avg_data.csv', 'Time [ms]', 'Average inference time', 'avg_data')
    create_plot('max_data.csv', 'Time [ms]', 'Maximum inference time', 'max_data')
    create_plot('dpu_perf.csv', 'Workload [GOP/s]', 'DPU Performance', 'dpu_perf')
    create_plot('mem_io.csv', 'Mem IO [MB]', 'Memory IO', 'mem_io')
    create_plot('mem_bandwth.csv', 'Mem Bandwidth [MB/s]', 'Memory Bandwidth', 'mem_bandwth')
