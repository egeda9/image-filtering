#include "itkMedianImageFilter.h"
#include "itkImage.h"
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"

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

  const auto input = itk::ReadImage<ImageType>(inputFileName);

  using FilterType = itk::MedianImageFilter<ImageType, ImageType>;
  auto medianFilter = FilterType::New();

  FilterType::InputSizeType radius;
  radius.Fill(radiusValue);

  medianFilter->SetRadius(radius);
  medianFilter->SetInput(input);

  try
  {
    itk::WriteImage(medianFilter->GetOutput(), outputFileName);
  }
  catch (const itk::ExceptionObject & error)
  {
    std::cerr << "Error: " << error << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}