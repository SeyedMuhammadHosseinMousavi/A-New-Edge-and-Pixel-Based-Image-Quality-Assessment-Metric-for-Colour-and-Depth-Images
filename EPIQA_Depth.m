%% 10 IQA metrics would be run on single depth image, which are:
% (1) The Peak Signal to Noise Ratio (PSNR)
% (2) The Signal to Noise Ratio (SNR) 
% (3) The Mean-Squared Error (MSE) 
% (4) The R-mean-squared error (RMSE) 
% (5) The Measure of Enhancement or Enhancement (EME) 
% (6) The Structural Similarity (SSIM) 
% (7) The Edge-strength Structural Similarity (ESSIM)
% (8) The Non-Shift Edge Based Ratio (NSER) 
% (9) The Edge Based Image Quality Assessment (EBIQA) 
% (10)The Edge and Pixel-Based Image Quality Assessment (EPIQA) 

%% 
clear;
%Reading
img=imread('Depth.jpg');
img=rgb2gray(img);%imshow(img);
%Adding Noise
saltnoise = imnoise(img,'salt & pepper', 0.1);
%Plot
% figure('units','normalized','outerposition',[0 0 1 1])
subplot(1,2,1);imshow(img);title('Ref');
subplot(1,2,2);imshow(saltnoise);title('Noise');
%PSNR and SNR between ref and polluted images
[peaksnr, snr] = psnr(saltnoise, img);
fprintf('\n(1)The Peak-SNR (PSNR) value is = %0.4f', peaksnr);
fprintf('\n(2) The (SNR) value is = %0.4f ', snr);
%MSE and RMSE between ref and polluted images
mseerr = immse(saltnoise, img);
rmseerr=sqrt(mseerr);
fprintf('\n(3) The mean-squared error (MSE) is = %0.4f', mseerr);
fprintf('\n(4) The R-mean-squared error (RMSE) is = %0.4f\n', rmseerr);
%% EME
% report format characters
newlineInAscii1 = [13 10];spaceInInAscii = 32;
% for printing, newline causes much confusion in matlab and is provided here as an alternative
newline = char(newlineInAscii1); spaceChar = char(spaceInInAscii);
% plot parameters
plotIndex = 1;plotRowSize = 1;plotColSize = 2;
% read the image
% targetFolder = 'images';
IMG = imread('man.jpg');  % IMG : originalImage
% IMG = strcat(targetFolder, '\', IMG);
% IMG = imread(IMG);
IMG = rgb2gray(IMG);IMG = double(IMG);
% noise parameters
sigma = 0.05;
offset = 0.01;
erosionFilterSize = 2;
dilationFilterSize = 2;
meann = 0;
noiseTypeModes = {
    'gaussian',         % [1]
    'salt & pepper',    % [2]    
    'localvar',         % [3]
    'speckle',          % [4] (multiplicative noise)
    'poisson',          % [5]
    'motion blur',      % [6]
    'erosion',          % [7]
    'dilation',         % [8]
    };
noiseChosen = 2;
noiseTypeChosen = char(noiseTypeModes(noiseChosen));
originalImage = uint8(IMG);
% plot original
titleStr = 'Original';
% imagePlot( originalImage, plotRowSize, plotColSize, ...
%                     plotIndex, titleStr );
% plotIndex = plotIndex + 1;
%
for i = 1:(plotRowSize*plotColSize)-1
IMG_aforeUpdated = double(IMG);   
% backup the previous state just in case it gets updated.
% returns the noise param updates for further corruption    
% IMG may be updated as the noisy image for the next round
[IMG, noisyImage, titleStr, sigma, dilationFilterSize, erosionFilterSize] = ...
    noisyImageGeneration(IMG, meann, sigma, offset, dilationFilterSize, erosionFilterSize, noiseTypeChosen);
imageQualityIndex_Value = imageQualityIndex(double(originalImage), double(noisyImage));
titleStr = [titleStr ',' newline 'IQI: ' num2str(imageQualityIndex_Value)];
% imagePlot( noisyImage, plotRowSize, plotColSize, ...
%                     plotIndex, titleStr );
plotIndex = plotIndex + 1;
end
if (~strcmp(char(class(noisyImage)), 'uint8'))
    disp('noisyImage is NOT type: uint8');
end
%% Calling measure of enhance- ment, or measure of improvement (EME)
[M M] = size(img);
L = 8;
EME_original = eme(double(img),M,L);
EME_noisyImage = eme(double(saltnoise),M,L);
fprintf('(5) (EME) (original image) = %5.5f \n', EME_original)
fprintf('(6) (EME) (noisy image) = %5.5f \n', EME_noisyImage)
    
%% SSIM and ESSIM
[ssimval, ssimmap] = ssim(saltnoise,img);
fprintf('(7) The (SSIM) value is = %0.4f.\n',ssimval)
[ESSIM_index] = ESSIM(img, saltnoise);
fprintf('(8) The (ESSIM) value is = %0.4f.\n',ESSIM_index)

%% NSER
% LOG filter
H = fspecial('log',10);
logfilter = imfilter(img,H,'replicate');
logfilternoise = imfilter(saltnoise,H,'replicate');% imshow(logfilter,[]);

% Zero crossing edge points
threshzero=[5 90];%it is possible to play with threshold
zerocrossing = edge(logfilter,'zerocross',threshzero);
zerocrossingnoise = edge(logfilternoise,'zerocross',threshzero);%imshow(zerocrossing,[]);

%NSE of two binary images
NSE=zerocrossing & zerocrossingnoise;% imshow(NSE,[]);

% log 10 of final image
logarithmfinal = log10(double(NSE));% imshow(logarithmfinal,[]);

%2-D correlation coefficient for final NSER
NSER = corr2(NSE,zerocrossing);
fprintf('(9) The (NSER) value is = %0.4f.\n',NSER)


%% EBIQA
% Resizing image (both images)
org = imresize(img, [256 256]);
noisy = imresize(saltnoise, [256 256]);

% Adding Sobel noise (both images)
org = edge(org,'Sobel');%imshow(org);
noisy = edge(noisy,'Sobel');%imshow(noisy);

% Dividing to 16*16 blocks (both images)
blocksorg=mat2tiles(org,[16,16]);
blocksnoisy=mat2tiles(noisy,[16,16]);

%EOI (Edge Orientation in Image): The number of edges which exists in each block
blocksorgsize=size(blocksorg);

for i=1:blocksorgsize(1,1)
   for j=1:blocksorgsize(1,2)
[labeledImage{i,j}, numberOfEdges{i,j}] = bwlabel(blocksorg{i,j});end;end;
for i=1:blocksorgsize(1,1)
   for j=1:blocksorgsize(1,2)
[labeledImagen{i,j}, numberOfEdgesn{i,j}] = bwlabel(blocksnoisy{i,j});end;end;

%PLE (Primitive Length of Edges): The number of pixels which exists in each block
for i=1:blocksorgsize(1,1)
   for j=1:blocksorgsize(1,2)
numberOfPixels{i,j} = nnz(blocksorg{i,j});end;end;
for i=1:blocksorgsize(1,1)
   for j=1:blocksorgsize(1,2)
numberOfPixelsn{i,j} = nnz(blocksnoisy{i,j});end;end;

%ALE (Average Length of the Edges): Lengths of edges in a block
for i=1:blocksorgsize(1,1)
   for j=1:blocksorgsize(1,2)
lengthOfEdges{i,j} = sum(blocksorg{i,j});end;end;
for i=1:blocksorgsize(1,1)
   for j=1:blocksorgsize(1,2)
lengthOfEdgesn{i,j} = sum(blocksnoisy{i,j});end;end;

% Average Length of the Edges
for i=1:blocksorgsize(1,1)
   for j=1:blocksorgsize(1,2)
averageEdgeLength{i,j} = lengthOfEdges{i,j} / numberOfEdges{i,j};end;end;
for i=1:blocksorgsize(1,1)
   for j=1:blocksorgsize(1,2)
averageEdgeLengthn{i,j} = lengthOfEdgesn{i,j} / numberOfEdgesn{i,j};end;end;

% NEP (Number of Edge Pixels): it is total number of pixels
totalnumberOfPixels= nnz(org);
totalnumberOfPixelsn= nnz(noisy);

% Euclidean distance claculation
for i=1:blocksorgsize(1,1)
   for j=1:blocksorgsize(1,2)
Euclideandiste(i,j) = pdist2(numberOfEdges{i,j},numberOfEdgesn{i,j});
Euclideandistp(i,j) = pdist2(numberOfPixels{i,j},numberOfPixelsn{i,j});
Euclideandistl(i,j) = pdist2(lengthOfEdges{i,j},lengthOfEdgesn{i,j});
Euclideandistt=pdist2(totalnumberOfPixels,totalnumberOfPixelsn);
end;end;
% Final matrix
Euclideandisfinal=[Euclideandiste Euclideandistp Euclideandistl];
% Average of final matrix (lower number shows higher quality)
EBIQA=mean(mean(Euclideandisfinal));
fprintf('(10) The (EBIQA) value is = %0.4f.\n',EBIQA)
%% Edge and Pixel-Based Image Quality Assessment Metric
EPIQA= (peaksnr)+ EBIQA;
fprintf('(11) The (EPIQA) value is = %0.4f.\n',EPIQA)
