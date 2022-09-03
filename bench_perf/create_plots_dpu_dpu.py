from matplotlib import pyplot as plt
import csv


def create_plot():
    dpu_thr_rows = []
    dpu_lat_rows = []

    with open('./thr_data_sets/dpu_results.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            else:
                dpu_thr_rows.append(row)

    with open('./lat_10k_test/dpu_results_lat.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            else:
                dpu_lat_rows.append(row)

    assert len(dpu_thr_rows) == len(dpu_lat_rows)

    x = []
    y_dpu_thr_time = []
    y_dpu_thr_thr = []
    y_dpu_lat_time = []
    y_dpu_lat_thr = []

    for i in range(len(dpu_thr_rows)):
        x.append(float(dpu_thr_rows[i][0]))
        y_dpu_thr_thr.append(round(float(dpu_thr_rows[i][1]), 1))
        y_dpu_thr_time.append(round(float(dpu_thr_rows[i][2]), 3))

        y_dpu_lat_thr.append(round(float(dpu_lat_rows[i][1]), 1))
        y_dpu_lat_time.append(round(float(dpu_lat_rows[i][2]), 3))

    # plot for time
    plt.figure()
    plt.xlabel('Parameters')
    plt.ylabel('Processing time [s]')

    plt.plot(x, y_dpu_thr_time, 'r', label='Throughput optimized')
    plt.plot(x, y_dpu_lat_time, 'b', label='Latency optimized')
    plt.legend()
    plt.title(' DPU Processing time comparison')

    plt.savefig('plots/dpu_comparison_time.png')

    # plot for throughput
    plt.figure()
    plt.xlabel('Parameters')
    plt.ylabel('Throughput [fps]')

    plt.plot(x, y_dpu_thr_thr, 'r', label='Throughput optimized')
    plt.plot(x, y_dpu_lat_thr, 'b', label='Latency optimized')
    plt.legend()
    plt.title('DPU Throughput comparison')

    plt.savefig('plots/dpu_comparison_throughput.png')


if __name__ == '__main__':
    create_plot()
