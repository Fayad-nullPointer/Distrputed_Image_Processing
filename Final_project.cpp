#include <opencv2/opencv.hpp>
#include <iostream>
#include <filesystem>
#include <mutex>
#include <omp.h>
#include <mpi.h>


using namespace std;
using namespace cv;
namespace fs = std::filesystem;

Mat applyColdFilter(const Mat& image);
Mat applyWarmFilter(const Mat& image);
Mat applyBWFilter(const Mat& image);
Mat applyFullBlur(const Mat& image, int ksize);
Mat applyLinearBlur(const Mat& image, int ksize);
Mat applyCircularBlur(const Mat& image, int ksize);
Mat resizeImage(const Mat& image, int newWidth, int newHeight);
Mat add_watermark(const Mat &image, const string &watermark_text = "Coffee",
                      Point position = cv::Point(170, 170),
                      double font_scale = 1.0,
                      double thickness = 2,
                      Scalar color = Scalar(5, 50, 55));
Mat segmentImage(Mat &image);


void processFolder(const string& folderPath, const string& outputFolder, const string& operation, int newWidth, int newHeight, string watermark_text);

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char folderPath[256] = {};
    char outputFolder[256] = {};
    char operation[10] = {};
    int newWidth = 0, newHeight = 0;
    string watermark_text = " ";

    if (rank == 0) {
        cout << "Enter the folder path containing images: ";
        cin.getline(folderPath, 256);
        cout << "Enter the output folder path: ";
        cin.getline(outputFolder, 256);
        cout << "Enter the operation (cold/warm/BW/blur/linear/circular/resize/segment/watermark): ";
        cin.getline(operation, 10);

        if (string(operation) == "watermark") {
            cout << "Enter the watermark text: ";
            cin >> watermark_text;
        }

        if (string(operation) == "resize") {
            cout << "Enter new width and height for resizing: ";
            cin >> newWidth >> newHeight;
        }
    }

    MPI_Bcast(folderPath, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(outputFolder, 256, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(operation, 10, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(&newWidth, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&newHeight, 1, MPI_INT, 0, MPI_COMM_WORLD);
    chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();

    processFolder(folderPath, outputFolder, operation, newWidth, newHeight,watermark_text);
    chrono::high_resolution_clock::time_point end = chrono::high_resolution_clock::now();
    chrono::duration<double> duration = chrono::duration_cast<chrono::duration<double>>(end - start);

    if (rank == 0) {
    cout << "Execution time: " << duration.count() << " seconds" << endl;
}
    MPI_Finalize();
    return 0;
}

Mat add_watermark(const Mat &image, const string &watermark_text,
                      Point position,
                      double font_scale,
                      double thickness ,
                      Scalar color ) {

    Mat watermarked_image = image.clone();

    int font = cv::FONT_HERSHEY_SIMPLEX;
    putText(watermarked_image, watermark_text, position, font, font_scale, color, thickness);
    return watermarked_image;
}



Mat segmentImage(Mat &image) {
    Mat gray, binary;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    threshold(gray, binary,0, 255, THRESH_BINARY+THRESH_OTSU);
    image = binary;
    return image;
}

Mat applyColdFilter(const Mat& image) {
    if (image.channels() == 1) return image;
    Mat coldImage;
    Mat channels[3];
    split(image, channels);
    channels[0] += 50;
    merge(channels, 3, coldImage);
    return coldImage;
}
Mat applyWarmFilter(const Mat& image) {
    if (image.channels() == 1) return image;
    Mat warmImage;
    Mat channels[3];
    split(image, channels);
    channels[2] += 50;
    merge(channels, 3, warmImage);
    return warmImage;
}


Mat applyBWFilter(const Mat& image) {
    Mat bwImage;
    cvtColor(image, bwImage, COLOR_BGR2GRAY);
    return bwImage;
}

Mat applyFullBlur(const Mat& image, int ksize) {
    Mat blurredImage;
    blur(image, blurredImage, Size(ksize, ksize));
    return blurredImage;
}

Mat applyLinearBlur(const Mat& image, int ksize) {
    Mat blurredImage;
    blur(image, blurredImage, Size(ksize, 1));
    return blurredImage;
}


Mat applyCircularBlur(const Mat& image, int ksize) {
    Mat blurredImage;
    GaussianBlur(image, blurredImage, Size(ksize, ksize), 0);
    return blurredImage;
}


Mat resizeImage(const Mat& image, int newWidth, int newHeight) {
    Mat resizedImage;
    if (!image.empty()) {
        resize(image, resizedImage, Size(newWidth, newHeight), 0, 0, INTER_LINEAR);
    }
    return resizedImage;
}

void processFolder(const string& folderPath, const string& outputFolder, const string& operation, int newWidth, int newHeight, string watermark_text) {
    vector<fs::path> imagePaths;
    std::mutex mtx;

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        if (entry.is_regular_file()) {
            imagePaths.push_back(entry.path());
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < static_cast<int>(imagePaths.size()); ++i) {
        string imagePath = imagePaths[i].string();
        Mat image = imread(imagePath);

        if (image.empty()) {
            cerr << "Could not open image: " << imagePath << endl;
            continue;
        }

        Mat processedImage;
        if (operation == "cold") {
            processedImage = applyColdFilter(image);
        } else if (operation == "warm") {
            processedImage = applyWarmFilter(image);
        } else if (operation == "BW") {
            processedImage = applyBWFilter(image);
        } else if (operation == "blur") {
            processedImage = applyFullBlur(image, 15);
        } else if (operation == "linear") {
            processedImage = applyLinearBlur(image, 15);
        } else if (operation == "circular") {
            processedImage = applyCircularBlur(image, 15);
        } else if (operation == "resize") {
            processedImage = resizeImage(image, newWidth, newHeight);
        }else if (operation == "segment") {
            processedImage = segmentImage(image);
        }else if (operation == "watermark") {
            processedImage = add_watermark(image, watermark_text);
        }else {
            cerr << "Invalid operation: " << operation << endl;
            continue;
        }

        string outputPath = outputFolder + "/" + imagePaths[i].filename().string();
        imwrite(outputPath, processedImage);

        #pragma omp critical
        {
            cout << "Processed and saved: " << outputPath << endl;
        }
    }
}
