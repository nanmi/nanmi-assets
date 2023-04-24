#include <spdlog/spdlog.h>

int main() {
    // Initialize logger
    auto logger = spdlog::basic_logger_mt("my_logger", "logs/my_log.txt");
    logger->set_level(spdlog::level::debug);

    // Log messages
    logger->debug("Debug message");
    logger->info("Info message");
    logger->warn("Warning message");
    logger->error("Error message");

    return 0;
}

// ===========================================

#include <cstdio>
#include <chrono>
#include <ctime>

void log(const std::string& message) {
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);

    std::FILE* file = std::fopen("logs/my_log.txt", "a");
    char* dt = std::ctime(&time);
    dt[strlen(dt)-1]=0;
    std::fprintf(file, "[%s] - %s\n", dt, message.c_str());
    std::fclose(file);
}

int main() {
    // Log messages
    log("Debug message");
    log("Info message");
    log("Warning message");
    log("Error message");

    return 0;
}

