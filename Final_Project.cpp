#include <iostream>
#include <filesystem>
#include <omp.h>
#include <mpi.h>
#include<opencv2/opencv.hpp>
using namespace std;
using namespace cv;
namespace fs = std::filesystem;
// Image segmentation function (for example: simple thresholding)
void segmentImage(Mat &image) {
    Mat gray, binary;
    cvtColor(image, gray, COLOR_BGR2GRAY);      // Convert to grayscale
    threshold(gray, binary,0, 255, THRESH_BINARY+THRESH_OTSU); // Outu's threshold thresholding
    image = binary;                             // Replace the original image with the segmented one
}

// Function to process all images in the given folder
void Image_Segmentation(const string &folderPath,const string &DitnationPath) {
     auto entry=nullptr;
    // Iterate over all files in the directory
    #pragma omp parallel shared(folderPath) private(entry)
    {
        int id=omp_get_thread_num();
    for (auto &entry: fs::directory_iterator(folderPath)) {
        string filePath = entry.path().string();
        // Read the image
        Mat image = imread(filePath);
        if (image.empty()) {
            cout << "Could not read image: " <<filePath << endl;
            continue;
        }

        // Perform segmentation on the image
        segmentImage(image);

        // Save the segmented image with a new name
        string outputFilePath =DitnationPath+"Segmented_"+entry.path().filename().string();
        imwrite(outputFilePath, image);
        cout << "Segmented image saved to: " << outputFilePath << endl;
    }
    }
}

int main(int argc, char** argv) {

    // Initialize MPI
    MPI_Init(&argc, &argv);

    // Get the rank and size for parallel processing
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // Only the root process will ask for the folder path
        cout << "Enter Folder of Images: ";
        string folderPath;
        getline(cin, folderPath); // Use getline for user input
        cout << "Enter Distantion of Images: ";
        string Distnation;
        getline(cin, Distnation);

        // Call the segmentation function
        Image_Segmentation(folderPath,Distnation);
    }

    // Finalize MPI
    MPI_Finalize();

    return 0;
}

