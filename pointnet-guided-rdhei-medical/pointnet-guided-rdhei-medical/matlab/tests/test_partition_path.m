clear; clc;

addpath(genpath(fullfile(pwd,'..'))); % ensure matlab/ is on path

xyz_path = fullfile('..','..','data','samples','demo.xyz');
lab_path = fullfile('..','..','data','samples','demo.label');

[P_int, meta] = read_xyz_quantize(xyz_path);
labels = read_label(lab_path);

cfg = struct();
cfg.startRule = 'minIndex'; % deterministic

subclouds = build_subclouds(labels);
fprintf('Number of subclouds: %d\n', numel(subclouds));

% Take the first subcloud
idx = subclouds(1).idx;
start_idx = choose_startpoint(idx, cfg);

path = nn_greedy_path(P_int, idx, start_idx);

fprintf('Subcloud label = %d, size = %d\n', subclouds(1).label, numel(idx));
fprintf('Start idx = %d\n', start_idx);
disp(path(1:min(10,end))');

