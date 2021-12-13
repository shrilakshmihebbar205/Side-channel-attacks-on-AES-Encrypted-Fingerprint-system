
%below code is gui initialization code for attack selection page
function varargout = MainCode(varargin)
% varargout : variable argument output- enables function to return sny
% number of output arguments
%varargin: variable argument input- it is an input variable in a function
%definition, it enables the function to accept any number of input
%arguments
gui_Singleton = 1;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @MainCode_OpeningFcn, ...
                   'gui_OutputFcn',  @MainCode_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
               
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end
%nargin : number of arguments in- returns negative of number of input arguments that
%appear in the function since there is varargin
%nargout :  negative of number of outputs that appear in function since
%there is varargout
if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end

%below code is the function to open gui page
%Executes just before MainCode is made visible. 
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data 
% handles is a function datatype : passes on one function to another 
% varargin  gives command line arguments to MainCode 
function MainCode_OpeningFcn(hObject, eventdata, handles, varargin)
% Choose default command line output for MainCode
handles.output = hObject;
warning off;
% Update handles structure
guidata(hObject, handles);

%below code is the output function to display maincode output on command
%window
%Outputs from this function are returned to the command line.
% varargout is a cell array for returning output args 
function varargout = MainCode_OutputFcn(hObject, eventdata, handles) 
% Get default command line output from handles structure
varargout{1} = handles.output;


%below code is for buttons on gui, 5 buttons namely
%radiobutton1 : EXISTING METHOD
%radiobutton2 : PROPOSED METHOD
%radiobutton3 : TIMING ATTACK
%radiobutton5 : CACHE ATTACK
%radiobutton6 : TIMING AND CACHE ATTACK
%sets the value corresponding to particular attack as 1, rest as zero
%hObject    handle to radiobutton
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data 
%get(hObject,'Value') returns toggle state of radiobutton
%Executes on button press in radiobutton2
function radiobutton1_Callback(hObject, eventdata, handles)
set(handles.radiobutton1,'Value',1)
set(handles.radiobutton2,'Value',0)

function radiobutton2_Callback(hObject, eventdata, handles)
set(handles.radiobutton1,'Value',0)
set(handles.radiobutton2,'Value',1)

function radiobutton3_Callback(hObject, eventdata, handles)
set(handles.radiobutton3,'Value',1)
set(handles.radiobutton5,'Value',0)
set(handles.radiobutton6,'Value',0)

function radiobutton5_Callback(hObject, eventdata, handles)
set(handles.radiobutton3,'Value',0)
set(handles.radiobutton5,'Value',1)
set(handles.radiobutton6,'Value',0)

function radiobutton6_Callback(hObject, eventdata, handles)
set(handles.radiobutton3,'Value',0)
set(handles.radiobutton5,'Value',0)
set(handles.radiobutton6,'Value',1)

%PUSH button is for start/perform button
function pushbutton1_Callback(hObject, eventdata, handles)
addpath('')

% global label
%ext_sel : existing method select
%prop_1 : proposed method
%get values from radio button and add it to variables

ext_sel = get(handles.radiobutton1,'Value');
prop_1= get(handles.radiobutton2,'Value');
tim_att = get(handles.radiobutton3,'Value');
cac_att = get(handles.radiobutton5,'Value');
all_att = get(handles.radiobutton6,'Value');

%open Toencrypt.txt : encryptdata1
% fget1 :returns next line of specified file, removing newline char
% tline: character vector with 1 row 1 column
fid = fopen('ToEncrypt.txt');
tline{1,1} = fgetl(fid);
count=1;
while ischar(tline)
    tline{count,1}= double(fgetl(fid));
    count=count+1;
end
fclose(fid);

%tline now has EncryptData1
%as long as it is character
%fill tline columns with the same text
% fclose closes the opened file
% encrypt_num = double(tline{1,1});

Input_dat= tline{1,1};
tic;
%tic : records the current time


imds = imageDatastore('Database', ...
    'IncludeSubfolders',true, ...
    'LabelSource','foldernames');
%image data store function: argument: location-database
%subfolder inclusion flag- true
%source providing the label data- foldernames : takes AUTHORIZED USER AND
%UNAUTHORIZED USER AS LABEL NAME


tbl=countEachLabel(imds);
%count each label: returns a table containing information about the labels and count for the
%input data stored in imds

minSetCount = min(tbl{:,2});
% minset count is 40- 40 items in each database

imds = splitEachLabel(imds,minSetCount,'randomize');
%split each label randomly: cleates imds object and labels each image according to
%the name of the folder it is in

[encrypt,key] = cryptosubcode(Input_dat);
%input_dat: EncryptData1 is given to cryptosubcode.m file, 
%it outputs an encrypted or encoded string, and a key that represents the
%random substituions made.

if (ext_sel==1)
    encrypted='fbhgsghb';
else
    encrypted= encrypt;
end
%if we select existing method, set encrypted to 'fbhgsghb'
%else it is random encoded string generated from cryptosubcode.m

encrypt_1 = char(encrypted);

%encrypted contains fbhgsghb
%encrypted.txt also contains fbhgsghb
% now fid contains fbhgsghb
fid = fopen('Encrypted.txt','wt');
fprintf(fid,encrypt_1);
fclose(fid);
%formated write to fid, the content in encrypt_1

%deep learning tool
net = resnet50();
%resnet50 is a convolution neural network that is 50 layers deep. 
%there is a database called ImageNet which has millions of images.
%using resnet50 function, we can load a pre trained version of network
%which is trained on Images from that database.
%this pre trained network helps in classification.
%output is returned to variable net.


analyzeNetwork(net);
%analyzenetwork function visualizes and understands the architecture of a
%network.
%detects problems like missing or unconnected layers before training
net.Layers(1);
%net.layers(i) defines the details of ith network layer
 
net.Layers(end);
numel(net.Layers(end).ClassNames);
%numel() - number of elements in the array of classnames
%classname array belongs to lastlayer


[trainingSet,testSet]=splitEachLabel(imds,0.7,'randomize');
%split each label randomly: cleates imds object and labels each image according to
%the name of the folder it is in

imageSize=net.Layers(1).InputSize;
%this pretrained model requires the image size to be same as inputsize of
%network. we use inputsize to determine for 1st layer.

augmentedTrainingSet = augmentedImageDatastore(imageSize,...
    trainingSet,'ColorPreprocessing','gray2rgb');

augmentedTestingSet = augmentedImageDatastore(imageSize,...
    testSet,'ColorPreprocessing','gray2rgb');
%augmented image datastore for testing and training
%this function transforms batches of training and testing data with optimal
%preprocessing such as resizing, rotation and reflection. to make them
%compatible with input size of deep learning network.


w1 = net.Layers(2).Weights;
w1=mat2gray(w1);
%gets weight of layer 2 and stores in w1 matrix
%converts matrix to grey scale image

figure(2);
montage(w1);
title('First Convolution Layer Weight');
%montage displays multiple image frames as rectangular montage
%display weights converted to greyscale to montage as single image

%feature extraction - check word file
featureLayer='fc1000_softmax';

trainingFeatures=activations(net,augmentedTrainingSet...
    ,featureLayer,'MiniBatchSize',12,'OutputAs','columns');
trainingLables = trainingSet.Labels;

classifier = fitcecoc(trainingFeatures, trainingLables...
    ,'Learner','Linear','Coding','onevsall'...
    ,'ObservationsIn','columns');

testFeatures=activations(net,augmentedTestingSet...
    ,featureLayer,'MiniBatchSize',12,'OutputAs','columns');

predictLabels = predict(classifier, testFeatures, 'ObservationsIn','columns');
testLabels=testSet.Labels;
%To automatically resize the training and test images before they are
%input to the network, create augmented image datastores, 
%specify the desired image size, and use these datastores as 
%input arguments to activations.


%To get the feature representations of the training and test images
%we use activations on the global average pooling layer 'fc1000_softmax'

confMat = confusionmat(testLabels, predictLabels);
%create confusion matrix
confMat = bsxfun(@rdivide, confMat, sum(confMat,2));
%bsxfun expands vectors into matrices of same size

Train_test_accurate=mean(diag(confMat))*100;
%record training time
train_time=toc;
[file,path]=uigetfile('*.*');
%uigetfile opens a dialog box that lists the files in that folder
%dialog box to select authorized or unauthorized folder
tic;
% tic function records current time
%toc function uses these recorded value to calculate elapsed time

%read fingerprint image to new image
I1=imread([path,file]);
newImage=I1;
I2=cat(3,newImage,newImage,newImage);
I2=imresize(I2,[280,320]);
I2=rgb2gray(I2);
figure('Name','INPUT AND PREPROCESS') 
imshow(uint8(I2));
title('INPUT IMAGE AND RESIZED IMAGE');
csvwrite('1.csv',I2);
newImage =imresize(newImage,[200,200]);
ImgInp= newImage;
% above code displays imput, resize image

%below code is to get encrypted image for corresponding input image
[n m k] = size(newImage);
n1 = n*m;
n1 = n1*8;
bin_x = zeros(n1,1,'uint8');
r = 3.9999998;
bin_x_N_Minus_1 =  0.300001;
x_N = 0;
tic
for ind = 2 : n1
    x_N = 1 - 2* bin_x_N_Minus_1 * bin_x_N_Minus_1;    
     if (x_N > 0.0)
        bin_x(ind-1) = 1;
    end 
     bin_x_N_Minus_1 =  x_N;
     
end
toc
t = uint8(0);
key_enc = zeros(n1/8,1,'uint8');
for ind1 = 1 : n1/8    
    for ind2 = 1 : 8
    key_enc(ind1) = key_enc(ind1) + bin_x(ind2*ind1)* 2 ^ (ind2-1);
    end      
end

for ind = 1 : m    
    Fkey(:,ind) = key_enc((1+(ind-1)*n) : (((ind-1)*n)+n));
end
len = n;
bre = m;
for ind = 1 : k
    Img = ImgInp(:,:,ind);
for ind1 = 1 : len
    for ind2 = 1 : bre        
        proImage(ind1,ind2) = bitxor(Img(ind1,ind2),Fkey(ind1,ind2));        
    end
end
proImageOut(:,:,ind) = proImage(:,:,1);
end
[n m k] = size(ImgInp);
for ind = 1 : m    
    Fkey(:,ind) = key_enc((1+(ind-1)*n) : (((ind-1)*n)+n));
end
len = n;
bre = m;
for ind = 1 : k
    Img = ImgInp(:,:,ind);
for ind1 = 1 : len
    for ind2 = 1 : bre        
        proImage(ind1,ind2) = bitxor(Img(ind1,ind2),Fkey(ind1,ind2));        
    end
end
proImageOut(:,:,ind) = proImage(:,:,1);
end

%encrypted image output
encrypted_image = proImageOut;
figure(6)
imshow(proImageOut)
title('Encrypted Image');
encrypt_time=toc;
tic;
%record encrypted time

encrypt = char(proImageOut);

I=csvread('1.csv');
rmin=16;
rmax=30;
scale=1;
rows=size(I,1);
cols=size(I,2);
[X,Y]=find(I<110);
s=size(X,1);
for k=1:s %
    if (X(k)>rmin)&(Y(k)>rmin)&(X(k)<=(rows-rmin))&(Y(k)<=(cols-rmin))
            A=I((X(k)-1):(X(k)+1),(Y(k)-1):(Y(k)+1));
            M=min(min(A));
           if I(X(k),Y(k))~=M
              X(k)=NaN;
              Y(k)=NaN;
           end
    end
end
v=find(isnan(X));
X(v)=[];
Y(v)=[];
index=find((X<=rmin)|(Y<=rmin)|(X>(rows-rmin))|(Y>(cols-rmin)));
X(index)=[];
Y(index)=[];
N=size(X,1);
maxb=zeros(rows,cols);
maxrad=zeros(rows,cols);

for j=1:N
    [b,r,blur]=partiald(I,[X(j),Y(j)],rmin,rmax,'inf',600,'first');%coarse search
    maxb(X(j),Y(j))=b;
    maxrad(X(j),Y(j))=r;
end
[x1,y1]=find(maxb==max(max(maxb)));
cp=search(I,rmin,rmax,x1,y1,'first');%fine search
cp=cp/scale;
out=drawcircle(I,[cp(1),cp(2)],cp(3),600);

AR = bwarea(out);
%bwarea estimates the area of objects in binary image bw
PR = mean(mean(bwperim(mean(out))));

%graycomatrix function creates a grey level co occurance matrix(GLCM) from image
%by calculating how often a pixel with grey level value i occurs
%horizontally adjacent to a  pixel with value j.
% i and j are specified as offset
GLCM2 = graycomatrix(out,'Offset',[2 0;0 2]);
stats = GLCM_fea(GLCM2,0);
v1 = stats.autoc;
v2 = stats.contr;
v3= stats.corrm;
v4=stats.corrp;
v5=stats.cprom;
v6=stats.cshad;
v7=stats.dissi;
v8=stats.energ;
v9=stats.entro;
v10=stats.homom;
v11=stats.homop;
v12=stats.maxpr;
v13=stats.sosvh;
v14=stats.savgh;
v15=stats.svarh;
v16=stats.senth;
v17=stats.dvarh;
v18=stats.denth;
v19=stats.inf1h;
v20=stats.inf2h;
v21=stats.indnc;
v22=stats.idmnc;
GLCM_fea1 = [v1 v2 v3 v4 v5 v6 v7 v8 v9 v10 v11 v12 v13 v14 v15 v16 v17 v18 v19 v20 v21 v22];
Testfea = [AR PR GLCM_fea1];
figure('Name','INPUT FEATURE')    
t = uitable('Data', Testfea,'Position', [20 200 400 200]); 

featureLayer='fc1000_softmax';
imageSize=net.Layers(1).InputSize;
ds = augmentedImageDatastore(imageSize,...
    newImage,'ColorPreprocessing','gray2rgb');
imageFeatures=activations(net,ds...
    ,featureLayer,'MiniBatchSize',12,'OutputAs','columns');
label = predict(classifier, imageFeatures, 'ObservationsIn','columns');

%calculate the decrypted time
decrypt_time = toc;
sprintf('The loaded image belongs to %s',label)
[cm,X,Y,per,TP,TN,FP,FN,sens1,spec1,precision,recall,Jaccard_coefficient,...
        Dice_coefficient,acc1,AUC] = Performance_Analysis(testLabels, predictLabels);  
figure('Name','Performance')    
colnames = {'Accuracy', 'Sensitivity', 'Specificity','TP','TN','FP',...
        'FN','Precision','Recall','Jaccard coefficient','Dice','AUC','Outer','classify Er'};
    
t = uitable('Data', [acc1 sens1 spec1 TP TN FP FN precision recall Jaccard_coefficient ...
        Dice_coefficient,mean(AUC)/10,cp(3),100-acc1], 'ColumnName', colnames, ...
        'Position', [20 200 400 200]);  
    
%performance graph
figure('Name','Performance Graph'),    
    bar([acc1 sens1 spec1],0.5),    
    set(gca,'XTickLabel',{'Accuracy','Sensitivity','Specificity'});    
    grid on;    
    ylabel('Estimated value');    
    title('Performance Graph');
    
%existing method decrypting the image using crypto sub decode
if (ext_sel==1)
decrypt_dat = cryptosubdecode(encrypted,key);
decrypt = char(decrypt_dat);
fid = fopen('AttackerDecrypted.txt','wt');
fprintf(fid,decrypt);
fclose(fid);

%below code is to decrypt the encrypted image
disp('Attacker Successfully Recovered the encrypted data!!!')
[n m k] = size(encrypted_image);

for ind = 1 : m    
    Fkey(:,ind) = key_enc((1+(ind-1)*n) : (((ind-1)*n)+n));
end
len = n;
bre = m;
for ind = 1 : k
    Img = encrypted_image(:,:,ind);
for ind1 = 1 : len
    for ind2 = 1 : bre        
        proImage(ind1,ind2) = bitxor(Img(ind1,ind2),Fkey(ind1,ind2));        
    end
end
proImageOut(:,:,ind) = proImage(:,:,1);
end
figure(1)
imshow(proImageOut)
title('attacker recovered data');
else
    disp('Attacker Cannot Recover the encrypted data!!!')
    decrypt_dat = cryptosubdecode(encrypted,key);

    if(label=="AUTHORIZED USER")
decrypt = char(decrypt_dat);
fid = fopen('Decrypted.txt','wt');
fprintf(fid,decrypt);
fclose(fid);
%disp('Input image belongs to the Authorized user');
    else
        %disp('Input image belongs to the Unauthorized user');
    end
end
 
to_sz = min(size(Input_dat,2),size(decrypt_dat,2));
detect_t = double(Input_dat(1:to_sz))-double(decrypt_dat(1:to_sz));

[row_1,~]= find(detect_t==0);

if (abs(size(row_1,2)-to_sz)< to_sz/4)
    if (tim_att==1)
       disp('!!Timing attack is not successful!!'); 
    end
    if(cac_att==1)
       disp('!!Cache attack is not successful!!');
    end
    if (all_att==1)
       disp('!!Timing attack is not successful!!'); 
       disp('!!Cache attack is not successful!!');         
    end
else
    if (tim_att==1)
    disp('!!!!!Timing attack is successful!!!!!'); 
    figure('Name','Time Graph'),    
    bar([train_time encrypt_time decrypt_time],0.5),    
    set(gca,'XTickLabel',{'Training time','Encryption time','Decryption time'});    
    grid on;    
    ylabel('Time Taken');    
    title('Time Graph');
    disp('--------------------Training Time:---------------')
    disp(train_time);
    disp('-------------------Encryption Time:--------------')
    disp(encrypt_time);
    disp('-------------------Decryption Time:-------------')
    disp(decrypt_time);
    end
    if(cac_att==1)
    disp('!!!!!cache attack is successful!!!!!');
    fid = fopen('Input image Encrypted.txt','wt');
    fprintf(fid,encrypt);
    fclose(fid); 
    disp('------------------Cache Attack:-------------');
    disp(char(encrypt));
    end

    if (all_att==1)
       disp('!!!Timing attack is successful!!'); 
       disp('!!!Cache attack is successful!!');   
       fid = fopen('Input image Encrypted.txt','wt');
       disp('--------------------Cache Attack:-----------------');
       disp(char(encrypt));
       fprintf(fid,encrypt);
       fclose(fid); 
       figure('Name','Time Graph'),       
       bar([train_time encrypt_time decrypt_time],0.5),  
       set(gca,'XTickLabel',{'Train','Encrypt','Decrypt'});
       grid on;    
       ylabel('Time Taken');    
       title('Time Graph');       
       disp('--------------------Training Time:---------------------')
       disp(train_time);
       disp('---------------------Ecryption Time:-------------------')
       disp(encrypt_time);
       disp('---------------------Decryption Time:-------------------')
       disp(decrypt_time);
       
    end
end
  
