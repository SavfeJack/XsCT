% hchan 2023/08
% read XA from C-arm DICOM

clear all; clc;
A = mat2gray(dicomread('C:\Users\user\Desktop\temp\000022E9'));
B = mat2gray(dicomread('C:\Users\user\Desktop\temp\000022EE'));
a = imcomplement(uint8(A.*255));
b = imcomplement(uint8(B.*255));

% for i = 1:size(a, 1)
%     for j = 1:size(a, 2)
%         if (sqrt((i - 512)^2 + (j - 512)^2)) > 508
%             a(i, j) = 0;
%             b(i, j) = 0;
%         end
%     end
% end

a = imresize(histeq(a), [256, 256]);
b = imresize(histeq(b), [256, 256]);
imwrite(imrotate(a, -90), 'C:\Users\user\Desktop\resized_01.bmp')
imwrite(imrotate(b, -90), 'C:\Users\user\Desktop\resized_02.bmp')