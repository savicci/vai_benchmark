from matplotlib import pyplot as plt
import csv


def create_plot():
    dpu_rows = []
    gpu_rows = []

    with open('./thr_data_sets/dpu_results.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            else:
                dpu_rows.append(row)

    with open('./gpu_data_sets/gpu_results.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            else:
                gpu_rows.append(row)

    assert len(dpu_rows) == len(gpu_rows)

    x = []
    y_dpu_time = []
    y_dpu_thr = []
    y_gpu_time = []
    y_gpu_thr = []

    for i in range(len(dpu_rows)):
        x.append(dpu_rows[i][0])
        y_dpu_thr.append(dpu_rows[i][1])
        y_dpu_time.append(dpu_rows[i][2])

        y_gpu_thr.append(gpu_rows[i][1])
        y_gpu_time.append(gpu_rows[i][2])

    # plot for time
    plt.figure()
    plt.xlabel('Parameters')
    plt.ylabel('Processing time [s]')

    plt.plot(x, y_dpu_time, 'r', label='DPU')
    plt.plot(x, y_gpu_time, 'b', label='GPU')
    plt.legend()
    plt.title('Processing time comparison')

    plt.savefig('plots/comparison_time.png')

    # plot for throughput
    plt.figure()
    plt.xlabel('Parameters')
    plt.ylabel('Throughput [fps]')

    plt.plot(x, y_dpu_thr, 'r', label='DPU')
    plt.plot(x, y_gpu_thr, 'b', label='GPU')
    plt.legend()
    plt.title('Throughput comparison')

    plt.savefig('plots/comparison_throughput.png')


if __name__ == '__main__':
    create_plot()
