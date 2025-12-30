function [H, used] = parse_header_bits(hb)
%PARSE_HEADER_BITS Parse the header from bit vector hb (0/1).
% Returns struct H and number of bits consumed.

    used = 0;

    H.MAGIC  = unpack_uint(hb(used+1:used+16)); used = used + 16;
    H.VER    = unpack_uint(hb(used+1:used+8));  used = used + 8;
    H.NBITS  = unpack_uint(hb(used+1:used+8));  used = used + 8;
    H.KMAX   = unpack_uint(hb(used+1:used+8));  used = used + 8;

    H.OFFSET = unpack_uint64(hb(used+1:used+64)); used = used + 64;

    H.N_UNITS  = unpack_uint(hb(used+1:used+32)); used = used + 32;
    H.LAB_BITS = unpack_uint(hb(used+1:used+32)); used = used + 32;
    H.MSG_BITS = unpack_uint(hb(used+1:used+32)); used = used + 32;
end

function v = unpack_uint(bits)
% bits MSB-first -> uint32
    n = numel(bits);
    v = uint32(0);
    for i = 1:n
        b = uint32(bits(i) ~= 0);
        v = bitshift(v,1) + b;
    end
end

function v = unpack_uint64(bits)
% bits MSB-first -> uint64
    n = numel(bits);
    v = uint64(0);
    for i = 1:n
        b = uint64(bits(i) ~= 0);
        v = bitshift(v,1) + b;
    end
end
