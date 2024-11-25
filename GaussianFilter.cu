//
// Created by sajit on 16/11/2024.
//


#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#pragma warning(disable : 4819)
#endif

#include <Exceptions.h>
#include <ImageIO.h>
#include <ImagesCPU.h>
#include <ImagesNPP.h>

#include <string>
#include <vector>
#include <iostream>
#include <filesystem>

#include <cuda_runtime.h>
#include <helper_cuda.h>

bool printfNPPinfo(int argc, char *argv[])
{
    const NppLibraryVersion *libVer = nppGetLibVersion();

    printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor,
           libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000,
           (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000,
           (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

void gaussFilter(const std::string &filePath, const std::string &outputFile)
{
    try
    {
        std::cout << "Processing of " << filePath << " started." << std::endl;
        npp::ImageCPU_8u_C1 hostSrc;
        npp::loadImage(filePath, hostSrc);
        npp::ImageNPP_8u_C1 deviceSrc(hostSrc);
        const NppiSize srcSize = {(int) deviceSrc.width(), (int) deviceSrc.height()};
        const NppiPoint srcOffset = {0, 0};

        const NppiSize filterROI = {(int) deviceSrc.width(), (int) deviceSrc.height()};
        npp::ImageNPP_8u_C1 deviceDst(filterROI.width, filterROI.height);

        NPP_CHECK_NPP(nppiFilterGaussBorder_8u_C1R(deviceSrc.data(), deviceSrc.pitch(), srcSize, srcOffset,
                                                   deviceDst.data(), deviceDst.pitch(), filterROI,
                                                   NppiMaskSize::NPP_MASK_SIZE_3_X_3,
                                                   NppiBorderType::NPP_BORDER_REPLICATE));

        npp::ImageCPU_8u_C1 hostDst(deviceDst.size());
        deviceDst.copyTo(hostDst.data(), hostDst.pitch());
        saveImage(outputFile, hostDst);
        std::cout << "Processing of " << filePath << " ended. Result saved to: " << outputFile << std::endl;

        nppiFree(deviceSrc.data());
        nppiFree(deviceDst.data());
        nppiFree(hostSrc.data());
        nppiFree(hostDst.data());
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
}

void applyFilter(const std::string &filterType, const std::string &filePath, const std::string &outputFile)
{

    std::cout << "Selected Gauss Smooth Filter." << std::endl;
    gaussFilter(filePath, outputFile);

    cudaDeviceSynchronize();
    cudaDeviceReset();
}

std::vector<std::string> splitString(const std::string &data, char separator)
{
    std::vector<std::string> strings;
    std::istringstream f(data);
    std::string s;
    while (getline(f, s, separator))
    {

        strings.push_back(s);
    }
    return strings;
}


int main(int argc, char *argv[])
{
    printf("Starting...\n\n");

    if (!printfNPPinfo(argc, argv))
    {
        exit(EXIT_SUCCESS);
    }

    findCudaDevice(argc, (const char **) argv);

    char *inputData;
    if (argc >= 2)
    {
        inputData = argv[1];
        if (!inputData)
        {
            std::cerr << "Cannot read the input data!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::cerr << "Input folder or image missed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    char *filterData;
    if (argc >= 3)
    {
        filterData = argv[2];
        if (!inputData)
        {
            std::cerr << "Cannot read the filter type!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::cerr << "Filter type is missed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    char *outputData;
    if (argc >= 4)
    {
        outputData = argv[3];
        if (!inputData)
        {
            std::cerr << "Cannot read the filter type!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        std::cerr << "Filter type is missed!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string filterType{filterData};
    std::string output{outputData};
    if (std::filesystem::is_directory(output))
    {
        std::filesystem::create_directory(output);
    }

    std::string inputValue{inputData};
    if (!std::filesystem::is_directory(inputValue))
    {
        std::string outputFile{output};
        if (std::filesystem::is_directory(output))
        {
            const std::string fileName = std::filesystem::path(inputValue).filename().string();
            const auto parts = splitString(fileName, '.');
            outputFile += "/" + parts.front() + ".bmp";
        }
        applyFilter(filterType, inputValue, outputFile);
    }
    return 0;
}