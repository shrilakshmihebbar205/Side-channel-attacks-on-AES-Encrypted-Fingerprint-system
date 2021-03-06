
function [b,r,blur]=partiald(I,C,rmin,rmax,sigma,n,part);
R=rmin:rmax;
count=size(R,2);
for k=1:count
[L(k)]=lineint(I,C,R(k),n,part);%computing the normalized line integral for each radius
if L(k)==0%if L(k)==0(this case occurs iff the radius takes the circle out of the image)
    %In this case,L is deleted as shown below and no more radii are taken for computation
    %(for that particular centre point).This is accomplished using the break statement
     L(k)=[];
    break;
end
end
D=diff(L);
D=[0 D];
%append one element at the beginning to make it an n vector
%Partial derivative at rmin is assumed to be zero
if strcmp(sigma,'inf')==1%the limiting case of the gaussian with sigma infinity(pls remember to change the code)strcmp syntax is different
f=ones(1,7)/7;
else
f=fspecial('gaussian',[1,5],sigma);%generates a 5 member 1-D gaussian
end
blur=convn(D,f,'same');%Smooths the D vecor by 1-D convolution 
%'same' indicates that size(blur) equals size(D)
blur=abs(blur);
[b,i]=max(blur);
r=R(i);
b=blur(i);
%calculates the blurred partial derivative
