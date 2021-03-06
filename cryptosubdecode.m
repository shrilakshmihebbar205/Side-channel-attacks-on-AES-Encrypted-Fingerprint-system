%decode function
%input arguments are given by cryptosubcode.m file
%input arguments: encoded string that must be decoded now, 
%and key representing random substitutions made.

function out = cryptosubdecode(in,key)

numbers = double(lower(in))-96;
decodednumbers = zeros(1,length(numbers));
for i = 1:length(numbers)
    if numbers(i) >= 1 & numbers(i) <= 26
        decodednumbers(i) = find(key==numbers(i));
    else
        decodednumbers(i) = numbers(i);
    end
end

out = char(decodednumbers+96);