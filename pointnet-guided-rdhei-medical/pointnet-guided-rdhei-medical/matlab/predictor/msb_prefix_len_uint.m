function L = msb_prefix_len_uint(a, b, nbits)
%MSB_PREFIX_LEN_UINT
% Return number of identical MSB bits between nonnegative integers a and b.
% a, b: uint64 scalars (or double convertible)
% nbits: fixed bit-width, e.g., 32 or 64
%
% L in [0, nbits]. If a==b, L = nbits.

    % ensure uint64
    a = uint64(a);
    b = uint64(b);

    x = bitxor(a, b);          % differing bits: 1 where different
    if x == 0
        L = nbits;
        return;
    end

    % Find position (0-based from LSB) of highest set bit in x
    % Old MATLAB friendly loop (nbits <= 64)
    msbPos = -1;
    for k = nbits-1:-1:0
        if bitget(x, k+1) % bitget index is 1-based
            msbPos = k;
            break;
        end
    end

    % If highest differing bit is at position msbPos (from LSB),
    % then identical MSB prefix length is:
    % L = nbits - 1 - msbPos
    L = (nbits - 1 - msbPos);
end


