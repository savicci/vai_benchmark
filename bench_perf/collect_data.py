import csv
import os
import argparse

# 10 columns + 1 empty
profile_summary_columns_num = 11

columns = ['Min time [ms]', 'Avg time [ms]', 'Max time [ms]', 'Performance [GOP/s]', 'Mem IO [Mb]',
           'Mem bandwidth  [Mb/s]']


def append_data_to_file(file_name, start_idx):
    header_row = ['Params', 'Batch4', 'Batch5', 'Batch5']
    data_row = [params]

    while start_idx < len(summary_rows):
        data_row.append(summary_rows[start_idx])
        start_idx += profile_summary_columns_num

    # write header row
    if not os.path.exists(file_name):
        with open(file_name, 'w') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header_row)

    with open(file_name, 'a') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(data_row)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-l', '--layer', type=int, default=1,
                        help='Layer to use for params')

    args = parser.parse_args()
    print('Command line options:')
    print(' --layer            : ', args.layer)

    with open('/workspace/vai_benchmark/bench_perf/params_{}.txt'.format(args.layer), 'r') as f:
        params = f.readline()

    summary_rows = []
    saving = False

    # read data
    with open('/workspace/vai_benchmark/bench_perf/profile_summary.csv', 'r') as file:
        reader = csv.reader(file, delimiter=',')
        for row in reader:
            if 'DPU Summary' in row:
                saving = True
                continue

            if saving:
                summary_rows.extend(row)


    # write data to files
    append_data_to_file('./min_data.csv', 14)
    append_data_to_file('./avg_data.csv', 15)

