function [H, used] = parse_header_bits_v2(hb)
%PARSE_HEADER_BITS_V2 Parse v2 header with LEN[].

    [H, used] = parse_header_bits(hb);

    H.S = uint8(unpack_uint(hb(used+1:used+8))); used = used + 8;

    H.LEN = zeros(double(H.S),1,'uint8');
    for i = 1:double(H.S)
        H.LEN(i) = uint8(unpack_uint(hb(used+1:used+8)));
        used = used + 8;
    end
end

function v = unpack_uint(bits)
    n = numel(bits);
    v = uint32(0);
    for i = 1:n
        b = uint32(bits(i) ~= 0);
        v = bitshift(v,1) + b;
    end
end
