%input:
%generates a coded string and a key from an input string
%input string given : toencrypt.txt : EncryptData1
%key represents the random substitutions made
%key is used further in cryptosubdecode.m file

%output:
%disp(coded); displays different encrypted key everytime
% eg: oeuvpnyfgyg1, zhpfsykoeke1
%disp(key);
%Columns 1 through 17
%  5     2    16    15    26    17    22    21    12     3    13     4    18     8     7    25    20
%Columns 18 through 26
%6    14    11     9    23    10    24    19     1

function [coded,key] = cryptosubcode(in)

[notused,key] = sort(rand(1,26));

numbers = double(lower(in)-96);
codednumbers = zeros(1,length(numbers));
for i = 1:length(numbers)
    if numbers(i) >= 1 & numbers(i) <= 26
        codednumbers(i) = key(numbers(i));
    else
        codednumbers(i) = numbers(i);
    end
end

coded = char(codednumbers+96);

