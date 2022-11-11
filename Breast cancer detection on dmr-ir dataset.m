%Pooya Iranmanesh

 clc
 clear all
 close all
 %DMR ir THERMOGRAPHY DATASET%%%C=cancer,N=normal
 Path='E:\total_dataset\*.jpg';     %place in which the dataset is held
 Files = dir(Path);   
%  feature_lbp=zeros(length(Files),59);
%  feature_hog=zeros(length(Files),167796);
 for i=1:length(Files)   
     fn = [Path(1:end-5 ) Files(i,1).name];
     im = imread(fn);  
     im1=rgb2gray(im);   
     im2=histeq(im1);   %histogram equalization
     im3=medfilt2(im2,[3,3]);   %median filter 2 dimentions
	 im4=locallapfilt(im3, sigma, alpha);%local laplasian filter
     %%%%%%%feature extraction%%%%%%%
     im5=im2double(im3);   
     f1=mean(im5(:));   
     f2=var(im5(:));   
     f3=std(im5(:));   
     f4=max(im5(:));   
     f5=min(im5(:));  
     f6=entropy(im5(:)); 
     f7=kurtosis(im5(:));   
     f8=skewness(im5(:));
     feature_stat(i,:)=[f1 f2 f3 f4 f5 f6 f7 f8];
     feature_lbp(i,:)=extractLBPFeatures(im5);   %local binary pattern
     feature_hog(i,:)=extractHOGFeatures(im5);
 end
%  figure()   
%  imshow(im)   
%  title('Raw Image')   
%  figure()   
%  imshow(im1)   
%  title('Gray Scale Image')   
%  figure()   
%  imshow(im2)  
%  title('Image After Applying Histogram equalization')  
%  figure()   
%  imshow(im3)   
%  title('Image After Applying Median Filter')  
%  figure()   
%  imshow(im4)  
%  title('Image After Applying Local Laplasian Filter')   
%  imhist(im1)  
%  title('Gray Scale Histogram')   
%  figure()   
%  imhist(im2)   
%  title('Histogram After Applying Histogram equalization')  


input=feature.stat;
input=feature_lbp;
input=feature_hog;
output=[ones(100, 1);zeros(100,1)]; 
%%nnstart=Neural Network Start
%%copy from nnstart -> simple script
x = input';
t = output';

Choose a Training Function
For a list of all training functions type: help nntrain
'trainlm' is usually fastest.
'trainbr' takes longer but may be better for challenging problems.
'trainscg' uses less memory. Suitable in low memory situations.
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

Create a Pattern Recognition Network
hiddenLayerSize = 8;
net = patternnet(hiddenLayerSize, trainFcn);

Setup Division of Data for Training, Validation, Testing
net.divideParam.trainRatio = 80/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 5/100;

Train the Network
[net,tr] = train(net,x,t);

Test the Network
y = net(x);
e = gsubtract(t,y);
performance = perform(net,t,y);
tind = vec2ind(t);
yind = vec2ind(y);
percentErrors = sum(tind ~= yind)/numel(tind);

View the Network
view(net)

Plots
Uncomment these lines to enable various plots.
%figure, plotperform(tr)
%figure, plottrainstate(tr)
%figure, ploterrhist(e)
figure, plotconfusion(t,y)
%figure, plotroc(t,y)