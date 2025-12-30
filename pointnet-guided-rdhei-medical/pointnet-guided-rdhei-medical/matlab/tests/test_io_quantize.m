clear; clc;

xyz_path = fullfile('..','..','data','samples','demo.xyz');
lab_path = fullfile('..','..','data','samples','demo.label');

[P_int, meta] = read_xyz_quantize(xyz_path);
labels = read_label(lab_path);

fprintf('Loaded points: %d\n', size(P_int,1));
fprintf('Quantization decimals d = %d, scale S = %lld\n', meta.decimals, meta.scale);

if size(P_int,1) ~= numel(labels)
    error('Mismatch: xyz has %d points, label has %d lines', size(P_int,1), numel(labels));
end

fprintf('OK: label count matches.\n');

% Quick sanity: show first 5 points (int)
disp(P_int(1:min(5,end), :));
disp(labels(1:min(5,end)));

