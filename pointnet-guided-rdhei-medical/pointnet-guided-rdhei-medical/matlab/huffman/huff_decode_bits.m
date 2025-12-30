function [symbols, usedBits] = huff_decode_bits(bits, dm, nsymbols)
%HUFF_DECODE_BITS Decode nsymbols from bitstream bits using decode maps.
% bits: uint8 0/1 vector, MSB-first
% returns symbols (nsymbols x 1) in 0..S-1 and usedBits count.

    bits = bits(:)';
    p = 1;
    symbols = zeros(nsymbols,1,'uint16');

    for i = 1:nsymbols
        code = uint32(0);

        found = false;
        for L = 1:dm.maxLen
            if p > numel(bits)
                error('Bitstream ended early while decoding.');
            end
            code = bitshift(code,1) + uint32(bits(p)~=0);
            p = p + 1;

            codesL = dm.lenList{L};
            if ~isempty(codesL)
                % find match
                j = find(codesL == code, 1, 'first');
                if ~isempty(j)
                    symbols(i) = dm.symList{L}(j);
                    found = true;
                    break;
                end
            end
        end
        if ~found
            error('Failed to decode symbol %d.', i);
        end
    end

    usedBits = p - 1;
end
