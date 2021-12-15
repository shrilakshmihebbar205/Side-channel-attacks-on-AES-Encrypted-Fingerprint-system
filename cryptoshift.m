%implements monoalphabetic shifting substitution encryption
%input arguments: string to be encoded, amount to shift each letter
%to decode this, we have to shift backwards th same number we shifted
%forward.
function out = cryptoshift(in,shift)
numbers = double(lower(in))-96;
codednumbers = zeros(1,length(numbers));
for i = 1:length(numbers);
    if numbers(i) >= 1 & numbers(i) <= 26
        codednumbers(i) = mod(numbers(i)+shift-1,26)+1;
    else
        codednumbers(i) = numbers(i);
    end
end

out = char(codednumbers+96);

function ciphertext = aes_encryption(plaintext,round_keys)
global m prim_poly fixM;
encrypt = 'e';
r = 0;
%plaintext input
plaintext_dec = double(plaintext);
input = bitxor(plaintext_dec,reshape(round_keys(:,:,r+1)', [1 16]));
for r = 1:10
% Byte substitution
    out_byte = byte_subs(input, encrypt);
    out_byte = reshape(out_byte, [4,4]);
     
% ShiftRows
    for i = 1:3
        out_byte(i+1,:) = circshift(out_byte(i+1,:),[0,3-((i+1)-2)]);
    end
    
% MixColumn
    if (r >= 1 && r <= 9)
        C = gf(fixM,m,prim_poly) * gf(out_byte,m,prim_poly);
        C = gf2dec(C,8,prim_poly);
    else
        C = reshape(out_byte, [1 16]);
    end
    
% Key Addition Layer
    input = bitxor(C,reshape(round_keys(:,:,r+1)', [1 16]));
end

% Ciphertext
ciphertext = input;

%decryption
function plaintext_recov = aes_decryption(ciphertext,round_keys)

global m prim_poly fixM_d;
decrypt = 'd'; 
r = 10; 
input = bitxor(ciphertext,reshape(round_keys(:,:,r+1)', [1 16]));
while(r >= 1 )
% MixColumn
    if (r <= 9 && r >= 1)
        C = reshape(input, [4 4]);
        B = gf(fixM_d,m,prim_poly) * gf(C,m,prim_poly);
        B = gf2dec(B,8,prim_poly);
    end
% Inv ShiftRows 
    if(r == 10)
        input = reshape(input, [4 4]);
        for i = 1:3
            input(i+1,:) = circshift(input(i+1,:),[0,i]);
        end
% Inv Byte substitution
        out_byte = byte_subs(reshape(input,[1 16]),decrypt);
    else
        B = reshape(B, [4 4]);
        for i = 1:3
            B(i+1,:) = circshift(B(i+1,:),[0,i]);
        end
%Inv Byte substitution
        out_byte = byte_subs(reshape(B,[1 16]),decrypt);
    end
% Key Addition Layer
    input = bitxor(out_byte,reshape(round_keys(:,:,(r-1)+1)', [1 16]));
    r = r - 1;
end
plaintext_recov = input;