#include <iostream>
#include <filesystem>

int main() {
    std::string path_str = "path/to/directory";
    std::filesystem::path path(path_str);

    if (!std::filesystem::exists(path)) {
        std::filesystem::create_directories(path);
        std::cout << "Directory created successfully" << std::endl;
    } else {
        std::cout << "Directory already exists" << std::endl;
    }

    return 0;
}
