function [L_map, is_anchor] = build_L_map(P_int, path, cfg)
%BUILD_L_MAP
% Build per-point MSB prefix length map L_map based on predecessor reference.
%
% Inputs:
%   P_int: (N x 3) int64 integer coordinates (can be negative)
%   path : (M x 1) indices (cover points to be predicted/embedded)
%   cfg.nbits: bit width to use (recommend 32 for scaled coords if safe, else 64)
%   cfg.offset: optional, nonnegative offset to shift coords to >=0 before uint conversion
%
% Outputs:
%   L_map: (M x 3) uint8, L_map(t,dim) is prefix length for point path(t)
%          relative to its predecessor path(t-1).
%          For t=1 (anchor), L_map(1,:) = 0 (but anchor is handled via is_anchor).
%   is_anchor: (M x 1) logical, true for t=1 (start point)
%
% Notes:
%   - We treat the first point as anchor (not embeddable, not predicted by others).
%   - L_map(1,:) is set to 0 for storage convenience, but you MUST use is_anchor
%     in later stages to distinguish anchor from true L=0 cases.

    if ~isfield(cfg,'nbits')
        cfg.nbits = 64;
    end
    nbits = cfg.nbits;

    % Offset to ensure nonnegative before uint conversion
    if isfield(cfg,'offset')
        offset = int64(cfg.offset);
    else
        % auto offset: shift by -min over used points if needed
        mins = min(P_int(path,:), [], 1);
        offset = int64(0);
        if any(mins < 0)
            offset = -min(mins); % make minimum 0 (use max shift among dims)
        end
    end

    M = numel(path);
    L_map = zeros(M, 3, 'uint8');
    is_anchor = false(M,1);
    is_anchor(1) = true;

    for t = 2:M
        cur = path(t);
        ref = path(t-1);

        for dim = 1:3
            a = uint64(P_int(ref, dim) + offset);
            b = uint64(P_int(cur, dim) + offset);

            L = msb_prefix_len_uint(a, b, nbits);
            if L > 255
                L = 255;
            end
            L_map(t, dim) = uint8(L);
        end
    end

    % anchor row left as zeros; use is_anchor to avoid ambiguity
end
