function [C_stego, n_written] = embed_msbprefix_stream(C_int, P_ref, path, L_map, is_anchor, bitstream, cfg)
%EMBED_MSBPREFIX_STREAM
% Embed bitstream into the MSB-identical prefix region (scheme A).
% Only embed into k bits within the first L bits (identical prefix).
%
% Inputs:
%   C_int: (N x 3) int64 encrypted coords (or just working coords)
%   P_ref: (N x 3) int64 reference coords for recovery (typically the SAME as C_int pre-embed,
%          and receiver uses recovered predecessor as reference during carrier recovery)
%   path, L_map, is_anchor: from previous steps
%   bitstream: row vector uint8(0/1)
%   cfg.nbits (default 64), cfg.kmax (default 8), cfg.offset (int64)
%
% Outputs:
%   C_stego: int64 stego coords
%   n_written: number of bits embedded

    if ~isfield(cfg,'nbits'), cfg.nbits = 64; end
    if ~isfield(cfg,'kmax'),  cfg.kmax  = 8;  end
    if ~isfield(cfg,'offset'), cfg.offset = int64(0); end

    nbits = cfg.nbits;
    kmax  = cfg.kmax;
    offset = int64(cfg.offset);

    C_stego = C_int;
    n_written = 0;

    bs = bitstream(:)'; % row
    p = 1; % pointer

    M = numel(path);
    for t = 1:M
        if is_anchor(t)
            continue;
        end
        cur = path(t);
        ref = path(t-1); % predecessor

        for dim = 1:3
            L = double(L_map(t,dim));
            if L <= 0
                continue;
            end

            k = min([kmax, L, numel(bs)-p+1]); % embed at most remaining bits
            if k <= 0
                continue;
            end

            % Positions within uint64 (0-based):
            % identical prefix spans [nbits-L, ..., nbits-1]
            % we embed into the lowest k bits of that prefix: [nbits-L, ..., nbits-L+k-1]
            pos_lo = nbits - L; % 0-based

            x = uint64(C_stego(cur, dim) + offset);

            payload = bs(p:p+k-1); % 0/1, MSB/LSB order as stored in stream
            % We write payload in LSB-first order into increasing bit positions.
            x2 = set_bits_range(x, pos_lo, fliplr(payload)); % make stream MSB-first -> place consistently
            C_stego(cur, dim) = int64(x2) - offset;

            p = p + k;
            n_written = n_written + k;

            if p > numel(bs)
                return;
            end
        end
    end
end
