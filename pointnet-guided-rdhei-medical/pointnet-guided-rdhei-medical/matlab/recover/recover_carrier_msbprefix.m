function C_rec = recover_carrier_msbprefix(C_stego, path, L_map, is_anchor, cfg)
%RECOVER_CARRIER_MSBPREFIX
% Recover original encrypted carrier (ignoring secret payload) by overwriting
% the contaminated MSB-prefix bits with reference point MSB-prefix bits.
%
% This is separable recovery: does NOT extract payload.

    if ~isfield(cfg,'nbits'), cfg.nbits = 64; end
    if ~isfield(cfg,'kmax'),  cfg.kmax  = 8;  end
    if ~isfield(cfg,'offset'), cfg.offset = int64(0); end

    nbits = cfg.nbits;
    kmax  = cfg.kmax;
    offset = int64(cfg.offset);

    C_rec = C_stego;

    M = numel(path);
    for t = 1:M
        if is_anchor(t)
            continue;
        end

        cur = path(t);
        ref = path(t-1); % predecessor, assumed already recovered

        for dim = 1:3
            L = double(L_map(t,dim));
            if L <= 0
                continue;
            end
            k = min(kmax, L);
            if k <= 0
                continue;
            end

            pos_lo = nbits - L; % where embedded bits were written
            x_cur = uint64(C_rec(cur, dim) + offset);
            x_ref = uint64(C_rec(ref, dim) + offset);

            % Overwrite the same k bits with reference bits
            ref_bits = get_bits_range(x_ref, pos_lo, k); % LSB-first
            x_new = set_bits_range(x_cur, pos_lo, ref_bits);

            C_rec(cur, dim) = int64(x_new) - offset;
        end
    end
end
