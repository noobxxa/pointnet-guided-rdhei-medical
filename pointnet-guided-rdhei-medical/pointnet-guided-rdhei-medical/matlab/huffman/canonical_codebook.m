function cb = canonical_codebook(lens)
%CANONICAL_CODEBOOK Build canonical Huffman codebook from code lengths.
% lens: (S x 1) uint8, 0 means unused
% cb: struct with fields:
%   .lens (Sx1)
%   .code (Sx1) uint32 code value (aligned to LSB)
%   .used (indices)
%   .maxLen
% Also provides a decode table builder separately.

    lens = double(lens(:));
    S = numel(lens);
    used = find(lens > 0);
    cb = struct();
    cb.lens = uint8(lens);
    cb.code = zeros(S,1,'uint32');
    cb.used = used;
    if isempty(used)
        cb.maxLen = 0;
        return;
    end
    maxLen = max(lens(used));
    cb.maxLen = maxLen;

    % Sort by (len, symbol)
    pairs = [lens(used), used];
    pairs = sortrows(pairs, [1 2]);

    code = uint32(0);
    prevLen = pairs(1,1);

    % first code has all zeros at its length
    cb.code(pairs(1,2)) = code;

    for i = 2:size(pairs,1)
        len = pairs(i,1);
        sym = pairs(i,2);

        code = code + 1;
        if len > prevLen
            code = bitshift(code, len - prevLen);
        end
        cb.code(sym) = code;
        prevLen = len;
    end
end
