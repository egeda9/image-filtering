#include "itkImage.h"
#include "itkImageFileReader.h"
#include <itkShapedNeighborhoodIterator.h>
#include <itkNeighborhood.h>
#include <itkImageRegionIterator.h>
#include "itkImageFileWriter.h"
#include "itkNeighborhoodAlgorithm.h"
#include <itkDiscreteGaussianImageFilter.h>
#include <itkStatisticsImageFilter.h>
#include <bits/stdc++.h>
using namespace std;

double getVariance(list<int> neighbors, const int mean) {
    const size_t sz = neighbors.size();
    if (sz <= 1) {
        return 0.0;
    }

    // Now calculate the variance
    auto variance_func = [&mean, &sz](int accumulator, const int& val) {
        return accumulator + ((val - mean)*(val - mean) / (sz - 1));
    };

    return std::accumulate(neighbors.begin(), neighbors.end(), 0.0, variance_func);
}

double getMean(list<int> neighbors) {
    const size_t sz = neighbors.size();
    if (sz <= 1) {
        return 0.0;
    }

    return std::accumulate(neighbors.begin(), neighbors.end(), 0.0) / sz;
}

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

    // read input image
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

    typedef itk::DiscreteGaussianImageFilter<ImageType, ImageType> GaussianFilterType;
    GaussianFilterType::Pointer gaussianFilter = GaussianFilterType::New();
    gaussianFilter->SetInput(reader->GetOutput());
    gaussianFilter->SetVariance(1.0);
    gaussianFilter->Update();

    typedef itk::StatisticsImageFilter<ImageType> StatisticsFilterType;
    StatisticsFilterType::Pointer statisticsFilter = StatisticsFilterType::New();
    statisticsFilter->SetInput(gaussianFilter->GetOutput());
    statisticsFilter->Update();
    float signalVariance = statisticsFilter->GetMean() - noiseVariance;

    // initialize output image
    ImageType::Pointer outputImage = ImageType::New();
    outputImage->CopyInformation(reader->GetOutput());
    outputImage->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
    outputImage->Allocate();
    //output->SetRegions(reader->GetOutput()->GetRequestedRegion());

    using ShapedNeighborhoodIteratorType = itk::ConstShapedNeighborhoodIterator<ImageType>;

    // define a radius for the shaped neighborhood iterator
    ShapedNeighborhoodIteratorType::RadiusType radius;
    radius.Fill(radiusValue);
    ShapedNeighborhoodIteratorType it(radius, reader->GetOutput(), reader->GetOutput()->GetLargestPossibleRegion());

    std::cout << "By default there are " << it.GetActiveIndexListSize() << " active indices." << std::endl;

    ShapedNeighborhoodIteratorType::OffsetType top = { { 0, -1 } };
    it.ActivateOffset(top);
    ShapedNeighborhoodIteratorType::OffsetType bottom = { { 0, 1 } };
    it.ActivateOffset(bottom);

    std::cout << "Now there are " << it.GetActiveIndexListSize() << " active indices." << std::endl;

    ShapedNeighborhoodIteratorType::IndexListType                 indexList = it.GetActiveIndexList();
    ShapedNeighborhoodIteratorType::IndexListType::const_iterator listIterator = indexList.begin();

    using IteratorType = itk::ImageRegionIterator<ImageType>;
    IteratorType out(outputImage, outputImage->GetRequestedRegion());

    // calculate mean and variance
    for (it.GoToBegin(), out.GoToBegin(); !it.IsAtEnd(); ++it, ++out)
    {
        ImageType::PixelType convolutionPixel = 0.0;

        std::cout << "New position: " << std::endl;
        ShapedNeighborhoodIteratorType::ConstIterator ci = it.Begin();

        list<int> neighbors;
        while (!ci.IsAtEnd())
        {
            convolutionPixel += ci.Get() * ci.Get();

            //neighbors.push_back(ci.Get());

            std::cout << "Centered at " << it.GetIndex() << std::endl;
            std::cout << "Neighborhood index " << ci.GetNeighborhoodIndex() << " is offset " << ci.GetNeighborhoodOffset()
                      << " and has value " << static_cast<unsigned>(ci.Get()) << ". The real index is "
                      << it.GetIndex() + ci.GetNeighborhoodOffset() << std::endl;
            ++ci;
        }

        convolutionPixel = convolutionPixel / it.GetRadius().size();
        ImageType::PixelType convolutionSquaredPixel = convolutionPixel * convolutionPixel;
        ImageType::PixelType alpha = noiseVariance / statisticsFilter->GetMean();
        ImageType::PixelType filterPixel = 1.0 / (1.0 + alpha * noiseVariance / convolutionSquaredPixel);
        ImageType::PixelType outputPixel = convolutionPixel * filterPixel;
        out.Set(outputPixel);
        ++it;
        ++out;

        //int mean = getMean(neighbors);
        //double variance = getVariance(neighbors, mean);
    }

    using WriterType = itk::ImageFileWriter<ImageType>;

    auto writer = WriterType::New();
    writer->SetFileName(argv[2]);
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