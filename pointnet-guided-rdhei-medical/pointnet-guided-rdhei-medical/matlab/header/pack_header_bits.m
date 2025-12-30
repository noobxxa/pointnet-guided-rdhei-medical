function hb = pack_header_bits(H)
%PACK_HEADER_BITS Pack a minimal header into a bit vector (0/1 uint8).
% Fields (fixed):
% MAGIC(16) VER(8) NBITS(8) KMAX(8)
% OFFSET(64) N_UNITS(32) LAB_BITS(32) MSG_BITS(32)
%
% Store integers in big-endian bit order per field (MSB-first within field).

    hb = uint8([]);

    hb = [hb, pack_uint(uint32(H.MAGIC), 16)];
    hb = [hb, pack_uint(uint32(H.VER),   8)];
    hb = [hb, pack_uint(uint32(H.NBITS), 8)];
    hb = [hb, pack_uint(uint32(H.KMAX),  8)];

    hb = [hb, pack_uint64(uint64(H.OFFSET), 64)];

    hb = [hb, pack_uint(uint32(H.N_UNITS), 32)];
    hb = [hb, pack_uint(uint32(H.LAB_BITS),32)];
    hb = [hb, pack_uint(uint32(H.MSG_BITS),32)];
end

function bits = pack_uint(v, nbits)
% Pack uint32 v into nbits bits, MSB-first.
    bits = zeros(1, nbits, 'uint8');
    for i = 1:nbits
        % take bit from MSB to LSB
        bpos = nbits - i; % 0-based within field
        bits(i) = uint8(bitget(uint64(v), bpos+1));
    end
end

function bits = pack_uint64(v, nbits)
% Pack uint64 v into nbits bits, MSB-first (nbits<=64).
    bits = zeros(1, nbits, 'uint8');
    for i = 1:nbits
        bpos = nbits - i;
        bits(i) = uint8(bitget(v, bpos+1));
    end
end
