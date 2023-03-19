#include "itkImage.h"
#include "itkImageFileReader.h"
#include <itkShapedNeighborhoodIterator.h>
#include <itkNeighborhood.h>
#include <itkImageRegionIterator.h>
#include "itkImageFileWriter.h"
#include <itkDiscreteGaussianImageFilter.h>
#include <itkStatisticsImageFilter.h>

using namespace std;

int main(int argc, char * argv[])
{
    if (argc != 5)
    {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0] << " <InputImageFile>" << std::endl;
        return EXIT_FAILURE;
    }

    constexpr unsigned int Dimension = 3;

    const char * inputFileName = argv[1];
    const char * outputFileName = argv[2];
    const int    radiusValue = std::stoi(argv[3]);
    const double noiseVariance = atof(argv[4]);

    using PixelType = unsigned char;
    using ImageType = itk::Image<PixelType, Dimension>;
    using ReaderType = itk::ImageFileReader<ImageType>;

    // Read input image
    ReaderType::Pointer  reader = ReaderType::New();
    reader->SetFileName(inputFileName);
    try
    {
        reader->Update();
    }
    catch (const itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

    // Initialize output image
    ImageType::Pointer outputImage = ImageType::New();
    outputImage->CopyInformation(reader->GetOutput());
    outputImage->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
    outputImage->Allocate();

    typedef itk::ShapedNeighborhoodIterator<ImageType> ShapedNeighborhoodIteratorType;

    // Define a radius for the shaped neighborhood iterator
    ShapedNeighborhoodIteratorType::RadiusType radius;
    radius.Fill(radiusValue);
    ShapedNeighborhoodIteratorType it(radius, reader->GetOutput(), reader->GetOutput()->GetLargestPossibleRegion());

    ShapedNeighborhoodIteratorType::OffsetType top = { { 0, -1 } };
    it.ActivateOffset(top);
    ShapedNeighborhoodIteratorType::OffsetType bottom = { { 0, 1 } };
    it.ActivateOffset(bottom);

    ShapedNeighborhoodIteratorType::IndexListType                 indexList = it.GetActiveIndexList();
    ShapedNeighborhoodIteratorType::IndexListType::const_iterator listIterator = indexList.begin();

    using IteratorType = itk::ImageRegionIterator<ImageType>;
    IteratorType out(outputImage, outputImage->GetRequestedRegion());

    // Loop pixels
    for (it.GoToBegin(), out.GoToBegin(); !it.IsAtEnd(); ++it, ++out)
    {
        // Algorithm and calculations

        // 1. Define a cubic neighborhood centered at (x,y,z) with a side length of N voxels
        // std::cout << "Centered at " << it.GetIndex() << std::endl;
        ImageType::IndexType index = it.GetIndex();
        ShapedNeighborhoodIteratorType::ConstIterator ci = it.Begin();

        // 2. Compute the local mean of the neighborhood
        // mean = (1/N^3) * sum of voxel values in neighborhood
        float mean = 0;
        for (ci.GoToBegin(); !ci.IsAtEnd(); ++ci) {
            mean += ci.Get();
        }
        mean /= index.size();

        // Compute the variance of the neighborhood
        // variance = (1/N^3) * sum of squared differences between voxel values and mean
        float variance = 0;
        for (ci.GoToBegin(); !ci.IsAtEnd(); ++ci) {
            float diff = ci.Get() - mean;
            variance += diff * diff;
        }
        variance /= index.size();

        // 3. Compute the filter coefficient for the voxel using the Wiener filter formula
        // w = variance / (variance + noise_variance)
        float voxel_filter_coef = variance / (variance + noiseVariance);

        // 4. Compute the filtered output value for the voxel using the weighted sum of the neighborhood
        // filtered_value = sum of (w * voxel value) in neighborhood
        double filteredValue = 0.0;
        for (ci.GoToBegin(); !ci.IsAtEnd(); ++ci) {
            filteredValue += (voxel_filter_coef * ci.Get());
        }

        // 5. Set the output image voxel at (x,y,z) to the filtered output value
        out.Set(filteredValue);
    }

    // Create output image
    using WriterType = itk::ImageFileWriter<ImageType>;

    auto writer = WriterType::New();
    writer->SetFileName(outputFileName);
    writer->SetInput(outputImage);
    try
    {
        writer->Update();
    }
    catch (const itk::ExceptionObject & err)
    {
        std::cerr << "ExceptionObject caught !" << std::endl;
        std::cerr << err << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}