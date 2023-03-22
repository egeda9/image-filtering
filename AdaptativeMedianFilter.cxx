#include <iostream>
#include <string>
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkNeighborhood.h"
#include "itkConstNeighborhoodIterator.h"
#include "itkNeighborhoodAlgorithm.h"
#include "itkStatisticsImageFilter.h"
#include "itkNeighborhoodAlgorithm.h"

int main(int argc, char * argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0] << " <InputImageFile> <OutputImageFile> <maxRadiusValue>" << std::endl;
        return EXIT_FAILURE;
    }

    using PixelType = unsigned char;
    constexpr unsigned int Dimension = 3;

    const char * inputFileName = argv[1];
    const char * outputFileName = argv[2];
    const int    maxRadiusValue = std::stoi(argv[3]);
    int          minRadiusValue = 1;

    using ImageType = itk::Image<PixelType, Dimension>;
    using ReaderType = itk::ImageFileReader<ImageType>;
    using WriterType = itk::ImageFileWriter<ImageType>;

    using NeighborhoodIteratorType = itk::ConstNeighborhoodIterator<ImageType>;

    auto reader = ReaderType::New();
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

    ImageType::Pointer outputImage = ImageType::New();
    outputImage->SetRegions(reader->GetOutput()->GetLargestPossibleRegion());
    outputImage->CopyInformation(reader->GetOutput());
    outputImage->Allocate();

    NeighborhoodIteratorType::RadiusType radius;
    radius.Fill(minRadiusValue);

    using IteratorType = itk::ImageRegionIterator<ImageType>;
    IteratorType out(outputImage, outputImage->GetRequestedRegion());

    NeighborhoodIteratorType iterator(radius, reader->GetOutput(),reader->GetOutput()->GetLargestPossibleRegion());

    for (iterator.GoToBegin(), out.GoToBegin(); !iterator.IsAtEnd(); ++iterator, ++out)
    {
        radius[0] = minRadiusValue;
        radius[1] = minRadiusValue;
        radius[2] = minRadiusValue;
        iterator.SetRadius(radius);
        bool repeatProcess = true;
        do
        {
            std::vector<PixelType> values;
            for (unsigned int i = 0; i < iterator.Size(); i++) {
                if (i != iterator.Size() / 2) {
                    values.push_back(iterator.GetPixel(i));
                }
            }
            std::sort(values.begin(), values.end());
            PixelType median = values[values.size() / 2];
            PixelType maxValue = *std::max_element(values.begin(), values.end());
            PixelType minValue = *std::min_element(values.begin(), values.end());
            PixelType centerValue = iterator.GetPixel(iterator.Size() / 2);
            if(minValue < median && median < maxValue)
            {
                repeatProcess = false;
                if(minValue < centerValue && centerValue < maxValue)
                {
                    out.Set(centerValue);
                }
                else
                {
                    out.Set(median);
                }
            }
            else
            {
                radius[0]++;
                radius[1]++;
                radius[2]++;
                iterator.SetRadius(radius);
            }
            if(iterator.GetRadius()[0] <= maxRadiusValue)
            {
                continue;
            }
            else
            {
                repeatProcess = false;
                out.Set(median);
            }
        } while(repeatProcess);
    }
    using WriterType = itk::ImageFileWriter<ImageType>;
    auto writer = WriterType::New();
    writer->SetFileName(outputFileName);
    writer->SetInput(outputImage);
    try    {
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

