function y = set_bits_range(x, pos_lo, bits)
%SET_BITS_RANGE Set bits of uint64 x starting at bit pos_lo (0-based).
% bits is 0/1 vector in LSB-first order (pos_lo corresponds to bits(1)).
    y = uint64(x);
    k = numel(bits);
    for i = 1:k
        y = bitset(y, pos_lo + i, bits(i) ~= 0);
    end
end
