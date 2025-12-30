clear; clc;
addpath(genpath(fullfile(pwd,'..')));

xyz_path = fullfile('..','..','data','samples','demo.xyz');
lab_path = fullfile('..','..','data','samples','demo.label');

[P_int, meta] = read_xyz_quantize(xyz_path);
labels = read_label(lab_path);

% Build path
cfg0 = struct(); cfg0.startRule = 'minIndex';
subclouds = build_subclouds(labels);
idx = subclouds(1).idx;
start_idx = choose_startpoint(idx, cfg0);
path = nn_greedy_path(P_int, idx, start_idx);

% Build L_map
pcfg = struct(); pcfg.nbits = 64;
[L_map, is_anchor] = build_L_map(P_int, path, pcfg);

% Offset (use same as build_L_map auto would; here keep 0 because your ints are nonnegative)
cfg = struct(); cfg.nbits = 64; cfg.kmax = 8; cfg.offset = int64(0);

% Create a dummy message bitstream
msg_len = 512;
msg = uint8(randi([0,1], 1, msg_len));

% Header
H = struct();
H.MAGIC = hex2dec('5244');
H.VER = 1;
H.NBITS = cfg.nbits;
H.KMAX = cfg.kmax;
H.OFFSET = uint64(cfg.offset);
H.N_UNITS = uint32((numel(path)-1)*3);
H.LAB_BITS = uint32(0); % placeholder for now
H.MSG_BITS = uint32(msg_len);

hb = pack_header_bits(H);

bitstream = [hb, msg];

% Embed into a "cipher" carrier (for now just use P_int as placeholder)
C0 = P_int;
[C_stego, n_written] = embed_msbprefix_stream(C0, C0, path, L_map, is_anchor, bitstream, cfg);
fprintf('Embedded bits: %d / %d\n', n_written, numel(bitstream));

% Extract back header+msg
ext = extract_msbprefix_stream(C_stego, path, L_map, is_anchor, numel(bitstream), cfg);
[H2, used] = parse_header_bits(ext);

fprintf('Parsed MAGIC = %04X, VER=%d, NBITS=%d, KMAX=%d, MSG_BITS=%d\n', ...
    H2.MAGIC, H2.VER, H2.NBITS, H2.KMAX, H2.MSG_BITS);

msg2 = ext(used+1:used+double(H2.MSG_BITS));
fprintf('Msg bit errors: %d\n', sum(msg2 ~= msg));

% Recover carrier ignoring secret
C_rec = recover_carrier_msbprefix(C_stego, path, L_map, is_anchor, cfg);
fprintf('Carrier recovery equal to original (int domain): %d\n', isequal(C_rec, C0));
