


def sliding_window_average(data, window_size):
    """
    Computes the sliding window average of a list of data points.

    Args:
        data (list): A list of data points.
        window_size (int): The size of the sliding window.

    Returns:
        list: A list of the sliding window averages.
    """
    if window_size > len(data):
        raise ValueError("Window size cannot be larger than data size.")
    window_sum = sum(data[:window_size])
    window_avg = window_sum / window_size
    averages = [window_avg]
    for i in range(window_size, len(data)):
        window_sum += data[i] - data[i-window_size]
        window_avg = window_sum / window_size
        averages.append(window_avg)
    return averages


if __name__ == '__main__':
    data = [0.1, 0.15, 0.05, 0.08, 0.07, 0.14, 0.12]
    window_size = 4
    data_res = sliding_window_average(data, window_size)

    for i in data_res:
        print(i)
