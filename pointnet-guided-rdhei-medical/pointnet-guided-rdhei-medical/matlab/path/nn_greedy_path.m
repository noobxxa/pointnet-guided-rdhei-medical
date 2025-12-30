function path = nn_greedy_path(P_int, idx, start_idx)
%NN_GREEDY_PATH Greedy nearest-neighbor path within a subcloud.
% Compatible with older MATLAB (no implicit expansion).

    idx = idx(:);
    if ~any(idx == start_idx)
        error('start_idx is not in idx.');
    end

    remaining = idx;
    path = zeros(numel(idx), 1);

    cur = start_idx;
    path(1) = cur;
    remaining(remaining == cur) = [];

    for t = 2:numel(path)
        cur_xyz = double(P_int(cur, :));      % 1x3
        rem_xyz = double(P_int(remaining, :));% Mx3

        % Old MATLAB: use bsxfun to subtract 1x3 from Mx3
        dif = bsxfun(@minus, rem_xyz, cur_xyz);  % Mx3
        dist2 = sum(dif.^2, 2);                  % Mx1

        [~, j] = min(dist2);
        cur = remaining(j);
        path(t) = cur;

        remaining(j) = [];
    end
end
