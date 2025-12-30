function bits = get_bits_range(x, pos_lo, k)
%GET_BITS_RANGE Extract k bits from uint64 x starting at bit pos_lo (0-based).
% Returns row vector bits(1..k) in LSB-first order (pos_lo is first).
    x = uint64(x);
    bits = zeros(1,k,'uint8');
    for i = 1:k
        bits(i) = uint8(bitget(x, pos_lo + i)); % bitget is 1-based
    end
end
