from matplotlib import pyplot as plt
import csv

cols_num = 4

if __name__ == '__main__':
    rows = []
    with open('./min_data.csv', 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for idx, row in enumerate(reader):
            if idx == 0:
                continue
            else:
                rows.extend(row)

    x = []
    y_batch4 = []
    y_batch5_first = []
    y_batch5_second = []
    for i in range(0, len(rows), cols_num):
        x.append(rows[i])
        y_batch4.append(rows[i + 1])
        y_batch5_first.append(rows[i + 2])
        y_batch5_second.append(rows[i + 2])

    # plot
    plt.figure()
    plt.subplot(1, 1, 1)
    plt.xlabel('Parameters')
    plt.plot(x, y_batch4, 'k')

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.plot(x, y_batch5_first, 'r--')

    plt.figure()
    plt.subplot(1, 1, 1)
    plt.plot(x, y_batch5_second, 'bo')

    plt.savefig('foo.png')
