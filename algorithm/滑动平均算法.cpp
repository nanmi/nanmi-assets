#include <vector>
#include <iostream>



std::vector<float> moving_average(const std::vector<float>& data, int window_size) {
    std::vector<float> smoothed_data(data.size());

    for (int i = 0; i < data.size(); i++) {
        float sum = 0;
        int count = 0;

        for (int j = i - window_size / 2; j <= i + window_size / 2; j++) {
            if (j >= 0 && j < data.size()) {
                sum += data[j];
                count++;
            }
        }

        smoothed_data[i] = sum / count;
    }

    return smoothed_data;
}


int main(int argc, char const *argv[])
{
    data = ...;
    int window_size = 9;
    std::vector<float> smoothed_data = moving_average(data, window_size);
    
    for (size_t i = 0; i < smoothed_data.size(); i++)
    {
        std::cout << smoothed_data.at(i) << std::endl;
    }
    
    return 0;
}
