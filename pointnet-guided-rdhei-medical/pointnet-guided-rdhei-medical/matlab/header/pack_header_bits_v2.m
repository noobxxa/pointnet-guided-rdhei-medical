function hb = pack_header_bits_v2(H)
%PACK_HEADER_BITS_V2 Header with canonical Huffman code lengths.
% Base fields + S(8) + LEN[0..S-1] each 8 bits.
% Fields:
% MAGIC16 VER8 NBITS8 KMAX8 OFFSET64 N_UNITS32 LAB_BITS32 MSG_BITS32
% S8 LEN[S]*8

    hb = pack_header_bits(H); % reuse v1 packer

    hb = [hb, pack_uint(uint32(H.S), 8)];
    for i = 1:H.S
        hb = [hb, pack_uint(uint32(H.LEN(i)), 8)];
    end
end

function bits = pack_uint(v, nbits)
    bits = zeros(1, nbits, 'uint8');
    for i = 1:nbits
        bpos = nbits - i;
        bits(i) = uint8(bitget(uint64(v), bpos+1));
    end
end
