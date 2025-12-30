
function subclouds = build_subclouds(labels)
%BUILD_SUBCLOUDS Group point indices by label.
% Input:
%   labels (N x 1) int64/int
% Output:
%   subclouds: struct array with fields:
%       .label : the label value
%       .idx   : indices of points in this subcloud (1-based)

    labels = labels(:);
    [ulab, ~, ic] = unique(labels, 'stable'); % old MATLAB friendly
    K = numel(ulab);

    subclouds = repmat(struct('label', [], 'idx', []), K, 1);

    for k = 1:K
        subclouds(k).label = ulab(k);
        subclouds(k).idx = find(ic == k);
    end
end
