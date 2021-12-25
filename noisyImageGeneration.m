function [IMG, noisyImage, titleStr, sigmaUpdated, dilationFilterSizeUpdated, erosionFilterSizeUpdated] = ...
    noisyImageGeneration(IMG, mean, sigma, offset, dilationFilterSize, erosionFilterSize, noiseType)
%NOISYIMAGEGENERATION 

newlineInAscii1 = [13 10];
spaceInInAscii = 32;
% for printing, newline causes much confusion in matlab and is provided here as an alternative
newline = char(newlineInAscii1); 
spaceChar = char(spaceInInAscii);

noiseTypeModes = {
    'gaussian',
    'salt & pepper',    
    'localvar',
    'speckle',
    'poisson',
    'motion blur',
    'erosion',
    'dilation',
    'jpg compression blocking effect'
    };

switch (lower(noiseType))
    case char(noiseTypeModes(1))
noisyImage = imnoise(uint8(IMG),'gaussian', mean, sigma);
            %adds Gaussian white noise of mean M and variance V to image I. When unspecified, M and V default to 0 and
            %0.01 respectively.
titleStr = ['Noisy image using Gaussian white noise', newline, 'with sigma =',num2str(sigma)];            
sigma = sigma + offset;   

    case char(noiseTypeModes(2))
noisyImage = imnoise(uint8(IMG), 'salt & pepper', sigma);
            %adds "salt and pepper" noise to the image I, where sigma is the noise density.  This affects approximately
            %sigma*numel(I) pixels. Default sigma = 0.05.
titleStr = ['Noisy image using Salt & Pepper noise', newline, 'with sigma =',num2str(sigma)]; 
sigma = sigma + offset;    

    case char(noiseTypeModes(3))    
imageMask = abs(sigma*randn(size(IMG))); % non-negative    
titleStr = ['Noisy image using Gaussian white noise ', newline, '(with noise mask) sigma =',num2str(sigma)]; 
    % J = imnoise(I,'localvar', V) adds zero-mean, Gaussian white noise of local variance, V, to the image I.  
    % V is an array of the same size as I.
noisyImage  = imnoise(uint8(IMG), 'localvar', imageMask);
sigma = sigma + offset;  

    case char(noiseTypeModes(4))
noisyImage  = imnoise(uint8(IMG), 'speckle', sigma);
            %adds multiplicative noise to the image I, using the equation J = I + n*I, where n is uniformly distributed random
            %noise with mean 0 and variance V. The default for V is 0.04.
titleStr = ['Noisy Image using multiplicative noise', newline, 'with sigma = ', num2str(sigma)];
sigma = sigma + offset;  

    case char(noiseTypeModes(5))
noisyImage  = imnoise(uint8(IMG), 'poisson');   
titleStr = ['Noisy Image using poisson'];
IMG = noisyImage;

    case char(noiseTypeModes(6))
PSF = fspecial('motion', 500*sigma, (500*sigma)-5); 
noisyImage = imfilter(IMG, PSF, 'symmetric','conv'); % 'conv', 'circular');
titleStr = ['Noisy Image using motion noise', newline, 'with sigma = ', num2str(sigma)];
sigma = sigma + offset; 

    case char(noiseTypeModes(7))  
erosionFilter = ones(erosionFilterSize, erosionFilterSize);
noisyImage = imerode(IMG, erosionFilter);
titleStr = ['Noisy image using Erosion with filter', newline, 'size =',num2str(erosionFilterSize)];
erosionFilterSize = erosionFilterSize + 1;

    case char(noiseTypeModes(8))
dilationFilter = ones(dilationFilterSize, dilationFilterSize);
noisyImage = imerode(IMG, dilationFilter);
titleStr = ['Noisy image using Dilation with filter', newline, 'size =',num2str(dilationFilterSize)];
dilationFilterSize = dilationFilterSize + 1;

    case char(noiseTypeModes(9))
       dilationFilterSize = 9;
outputFolder = 'outputFiles';
outputFileTarget = 'tmp.jpg';
mkdir(outputFolder);
outputFileTarget = strcat(outputFolder, '\', outputFileTarget);
imwrite(IMG/255, outputFileTarget, 'quality', dilationFilterSize);
noisyImage = double(imread(outputFileTarget));
titleStr = ['Noisy image using jpg compression blocking effect,', newline, 'blocking factor =',num2str(dilationFilterSize)];
% dilationFilterSize = dilationFilterSize + 1;
IMG = noisyImage;
delete(outputFileTarget);


end % END of noise type choice

sigmaUpdated = sigma;
dilationFilterSizeUpdated = dilationFilterSize;
erosionFilterSizeUpdated = erosionFilterSize;

end

