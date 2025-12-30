clear; clc;

addpath(genpath(fullfile(pwd,'..')));

xyz_path = fullfile('..','..','data','samples','demo.xyz');
lab_path = fullfile('..','..','data','samples','demo.label');

[P_int, meta] = read_xyz_quantize(xyz_path);
labels = read_label(lab_path);

cfg = struct();
cfg.startRule = 'minIndex';

% Build path (single subcloud for now)
subclouds = build_subclouds(labels);
idx = subclouds(1).idx;
start_idx = choose_startpoint(idx, cfg);
path = nn_greedy_path(P_int, idx, start_idx);

% Predictor config
pcfg = struct();
pcfg.nbits = 64;  % safe default; later we can tighten to 32 if range allows

[L_map, is_anchor] = build_L_map(P_int, path, pcfg);

% Compute statistics excluding anchor
Lm = double(L_map(~is_anchor, :));
fprintf('Points (excluding anchor): %d\n', size(Lm,1));
fprintf('Mean L (x,y,z): %.2f, %.2f, %.2f\n', mean(Lm(:,1)), mean(Lm(:,2)), mean(Lm(:,3)));
fprintf('Min  L (x,y,z): %.0f, %.0f, %.0f\n', min(Lm(:,1)), min(Lm(:,2)), min(Lm(:,3)));
fprintf('Max  L (x,y,z): %.0f, %.0f, %.0f\n', max(Lm(:,1)), max(Lm(:,2)), max(Lm(:,3)));

disp('First 5 rows of L_map (anchor row included):');
disp(L_map(1:5, :));
