function bits = huff_encode_symbols(symbols, cb)
%HUFF_ENCODE_SYMBOLS Encode a vector of symbols (0..S-1) using codebook cb.
% Output bits are uint8(0/1), MSB-first per codeword.

    symbols = symbols(:);
    bits = uint8([]);

    for i = 1:numel(symbols)
        s = symbols(i) + 1; % MATLAB index
        len = double(cb.lens(s));
        if len <= 0
            error('Symbol %d has zero code length.', symbols(i));
        end
        code = uint32(cb.code(s));

        % Emit MSB-first within this len
        word = zeros(1,len,'uint8');
        for j = 1:len
            bpos = len - j; % 0-based within len
            word(j) = uint8(bitget(uint64(code), bpos+1));
        end
        bits = [bits, word]; %#ok<AGROW>
    end
end
