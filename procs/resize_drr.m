% hchan 2023/08
% read XA from C-arm DICOM

clear all; clc;
a = imread('C:\Users\user\Desktop\temp\drr01.bmp');
b = imresize(a, [256, 256]);
% imshow(b)
imwrite(b, 'C:\Users\user\Desktop\temp\resized_drr01.bmp')

a2 = imread('C:\Users\user\Desktop\temp\drr02.bmp');
b2 = imresize(a2, [256, 256]);
% imshow(b2)
imwrite(b2, 'C:\Users\user\Desktop\temp\resized_drr02.bmp')