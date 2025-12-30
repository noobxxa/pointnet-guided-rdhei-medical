clear; clc;
addpath(genpath(fullfile(pwd,'..')));

xyz_path = fullfile('..','..','data','samples','demo.xyz');
lab_path = fullfile('..','..','data','samples','demo.label');

[P_int, meta] = read_xyz_quantize(xyz_path);
labels = read_label(lab_path);

% Path
cfg0 = struct(); cfg0.startRule = 'minIndex';
subclouds = build_subclouds(labels);
idx = subclouds(1).idx;
start_idx = choose_startpoint(idx, cfg0);
path = nn_greedy_path(P_int, idx, start_idx);

% L_map
pcfg = struct(); pcfg.nbits = 64;
[L_map, is_anchor] = build_L_map(P_int, path, pcfg);

% Params
cfg = struct(); cfg.nbits = 64; cfg.kmax = 8; cfg.offset = int64(0);

% Build k_map per unit (t,dim) in traversal order (excluding anchor)
% Here: k = min(kmax, L)
k_list = [];
unit_pos = []; % store (t,dim) if you want later; optional
for t = 1:numel(path)
    if is_anchor(t), continue; end
    for dim = 1:3
        L = double(L_map(t,dim));
        k = min(cfg.kmax, L);
        k_list(end+1,1) = uint8(k); %#ok<AGROW>
    end
end
N_units = numel(k_list);

% Huffman over symbols k in [0..kmax]
S = cfg.kmax + 1;
freq = zeros(S,1);
for i = 1:N_units
    freq(double(k_list(i))+1) = freq(double(k_list(i))+1) + 1;
end

lens = huffman_codelens(freq);
cb = canonical_codebook(lens);
dm = huff_build_decodemaps(cb);

% Encode label stream (k_list)
lab_bits = huff_encode_symbols(uint16(k_list), cb); % symbols are k
LAB_BITS = numel(lab_bits);

% Dummy secret message
MSG_BITS = 512;
msg = uint8(randi([0,1], 1, MSG_BITS));

% Header v2
H = struct();
H.MAGIC = hex2dec('5244');
H.VER = 2;
H.NBITS = cfg.nbits;
H.KMAX = cfg.kmax;
H.OFFSET = uint64(cfg.offset);
H.N_UNITS = uint32(N_units);
H.LAB_BITS = uint32(LAB_BITS);
H.MSG_BITS = uint32(MSG_BITS);
H.S = uint8(S);
H.LEN = uint8(lens); % length per symbol 0..kmax

hb = pack_header_bits_v2(H);

bitstream = [hb, lab_bits, msg];

% Embed
C0 = P_int;
[C_stego, n_written] = embed_msbprefix_stream(C0, C0, path, L_map, is_anchor, bitstream, cfg);
fprintf('Embedded bits: %d / %d\n', n_written, numel(bitstream));

% Extract back (all)
ext = extract_msbprefix_stream(C_stego, path, L_map, is_anchor, numel(bitstream), cfg);

% Parse header
[H2, used] = parse_header_bits_v2(ext);
fprintf('Parsed VER=%d, KMAX=%d, N_UNITS=%d, LAB_BITS=%d, MSG_BITS=%d, S=%d\n', ...
    H2.VER, H2.KMAX, H2.N_UNITS, H2.LAB_BITS, H2.MSG_BITS, H2.S);

% Rebuild codebook from LEN
lens2 = double(H2.LEN(:));
cb2 = canonical_codebook(uint8(lens2));
dm2 = huff_build_decodemaps(cb2);

% Decode labels
lab_start = used + 1;
lab_end = used + double(H2.LAB_BITS);
lab_bits2 = ext(lab_start:lab_end);

[k_dec, usedLab] = huff_decode_bits(lab_bits2, dm2, double(H2.N_UNITS));

% Extract msg
msg_start = lab_end + 1;
msg_end = lab_end + double(H2.MSG_BITS);
msg2 = ext(msg_start:msg_end);

fprintf('Label decode match: %d\n', isequal(uint8(k_dec), k_list));
fprintf('Msg bit errors: %d\n', sum(uint8(msg2) ~= msg));

% Recover carrier (ignoring secret)
C_rec = recover_carrier_msbprefix(C_stego, path, L_map, is_anchor, cfg);
fprintf('Carrier recovery equal to original (int domain): %d\n', isequal(C_rec, C0));
