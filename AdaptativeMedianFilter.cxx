#include "itkMedianImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include <itkShapedNeighborhoodIterator.h>
#include <itkNeighborhood.h>

int main(int argc, char * argv[])
{
    if (argc != 4)
    {
        std::cerr << "Usage: " << std::endl;
        std::cerr << argv[0] << " <InputImageFile> <OutputImageFile> <radius>" << std::endl;
        return EXIT_FAILURE;
    }

    constexpr unsigned int Dimension = 3;

    const char * inputFileName = argv[1];
    const char * outputFileName = argv[2];
    const int    radiusValue = std::stoi(argv[3]);

    using PixelType = unsigned char;
    using ImageType = itk::Image<PixelType, Dimension>;
    using ReaderType = itk::ImageFileReader<ImageType>;

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

    using NeighborhoodIteratorType = itk::ShapedNeighborhoodIterator<ImageType>;

    NeighborhoodIteratorType::RadiusType radius;
    radius.Fill(radiusValue);
    NeighborhoodIteratorType it(radius, reader->GetOutput(), reader->GetOutput()->GetLargestPossibleRegion());
    std::cout << "By default there are " << it.GetActiveIndexListSize() << " active indices." << std::endl;

    NeighborhoodIteratorType::OffsetType top = { { 0, -1 } };
    it.ActivateOffset(top);
    NeighborhoodIteratorType::OffsetType bottom = { { 0, 1 } };
    it.ActivateOffset(bottom);

    std::cout << "Now there are " << it.GetActiveIndexListSize() << " active indices." << std::endl;

    NeighborhoodIteratorType::IndexListType                 indexList = it.GetActiveIndexList();
    NeighborhoodIteratorType::IndexListType::const_iterator listIterator = indexList.begin();

    // Note that ZeroFluxNeumannBoundaryCondition is used by default so even
    // pixels outside of the image will have valid values (equivalent to
    // their neighbors just inside the image)
    for (it.GoToBegin(); !it.IsAtEnd(); ++it)
    {
        std::cout << "New position: " << std::endl;
        NeighborhoodIteratorType::ConstIterator ci = it.Begin();

        while (!ci.IsAtEnd())
        {
            std::cout << "Centered at " << it.GetIndex() << std::endl;
            std::cout << "Neighborhood index " << ci.GetNeighborhoodIndex() << " is offset " << ci.GetNeighborhoodOffset()
                      << " and has value " << ci.Get() << " The real index is "
                      << it.GetIndex() + ci.GetNeighborhoodOffset() << std::endl;
            ++ci;
        }
    }

    std::cout << std::endl;

    return EXIT_SUCCESS;
}