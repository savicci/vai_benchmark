from matplotlib import pyplot as plt
import csv

COLS_NUM = 4


def create_plot(filename, ylabel, title,  save_filename, op_type):
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

        if op_type == 'sum':
            y.append(float(rows[i + 1]) + float(rows[i + 2]) + float(rows[i + 2]))
        elif op_type == 'avg':
            y.append((float(rows[i + 1]) + float(rows[i + 2]) + float(rows[i + 2]))/3)
        else:
            print('Unknown op type: {}'.format(op_type))
            exit(-1)

    # plot
    plt.figure()
    plt.xlabel('Parameters')
    plt.ylabel(ylabel)

    plt.plot(x, y, 'r')
    plt.title(title)

    plt.savefig('plots/{}.png'.format(save_filename))
    # plt.show()


if __name__ == '__main__':
    create_plot('min_data.csv', 'Time [ms]', 'Minimal inference time', 'min_data', 'avg')
    create_plot('avg_data.csv', 'Time [ms]', 'Average inference time', 'avg_data', 'avg')
    create_plot('max_data.csv', 'Time [ms]', 'Maximum inference time', 'max_data', 'avg')
    create_plot('dpu_perf.csv', 'Workload [GOP/s]', 'DPU Performance', 'dpu_perf', 'sum')
    create_plot('mem_io.csv', 'Mem IO [MB]', 'Memory IO', 'mem_io', 'sum')
    create_plot('mem_bandwth.csv', 'Mem Bandwidth [MB/s]', 'Memory Bandwidth', 'mem_bandwth', 'sum')
