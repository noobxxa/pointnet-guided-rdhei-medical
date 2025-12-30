function bits = extract_msbprefix_stream(C_stego, path, L_map, is_anchor, nbits_total, cfg)
%EXTRACT_MSBPREFIX_STREAM Extract a given number of bits from scheme A.
% cfg.nbits, cfg.kmax, cfg.offset

    if ~isfield(cfg,'nbits'), cfg.nbits = 64; end
    if ~isfield(cfg,'kmax'),  cfg.kmax  = 8;  end
    if ~isfield(cfg,'offset'), cfg.offset = int64(0); end

    nbits = cfg.nbits;
    kmax  = cfg.kmax;
    offset = int64(cfg.offset);

    bits = zeros(1, nbits_total, 'uint8');
    p = 1;

    M = numel(path);
    for t = 1:M
        if is_anchor(t)
            continue;
        end
        cur = path(t);

        for dim = 1:3
            L = double(L_map(t,dim));
            if L <= 0
                continue;
            end

            k = min([kmax, L, nbits_total - p + 1]);
            if k <= 0
                continue;
            end

            pos_lo = nbits - L; % 0-based
            x = uint64(C_stego(cur, dim) + offset);

            got = get_bits_range(x, pos_lo, k); % LSB-first
            got = fliplr(got);                 % back to MSB-first stream order

            bits(p:p+k-1) = got;
            p = p + k;

            if p > nbits_total
                return;
            end
        end
    end
end
