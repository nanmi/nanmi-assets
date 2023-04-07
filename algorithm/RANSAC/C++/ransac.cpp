#include <vector>
#include <random>
#include <algorithm>

// Define a struct to hold the data points
struct Point {
    float x;
    float y;
};

// Define a function to calculate the distance between two points
float distance(Point p1, Point p2) {
    float dx = p1.x - p2.x;
    float dy = p1.y - p2.y;
    return std::sqrt(dx*dx + dy*dy);
}

// Define a function to fit a line to a set of points using RANSAC
std::vector<Point> ransac(std::vector<Point> points, int iterations, float threshold) {
    std::vector<Point> best_inliers;
    int best_count = 0;
    std::random_device rd;
    std::mt19937 gen(rd());
    for (int i = 0; i < iterations; i++) {
        // Randomly select two points
        std::uniform_int_distribution<> dis(0, points.size()-1);
        Point p1 = points[dis(gen)];
        Point p2 = points[dis(gen)];
        // Calculate the line between the two points
        float a = (p2.y - p1.y) / (p2.x - p1.x);
        float b = p1.y - a * p1.x;
        // Count the number of inliers
        std::vector<Point> inliers;
        for (Point p : points) {
            if (distance(p, Point{p.x, a*p.x+b}) < threshold) {
                inliers.push_back(p);
            }
        }
        // Update the best model if this one has more inliers
        if (inliers.size() > best_count) {
            best_inliers = inliers;
            best_count = inliers.size();
        }
    }
    return best_inliers;
}