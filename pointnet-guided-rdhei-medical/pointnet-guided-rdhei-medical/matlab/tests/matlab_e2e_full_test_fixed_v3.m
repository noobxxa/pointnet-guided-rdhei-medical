%% matlab_e2e_full_test_fixed_v3.m  (BRIDGE + CLEAN + RAW LABELS DEFAULT)
% This script runs an end-to-end test:
%  - Read sampled 8192-point xyz carrier from bridge
%  - Read 8192 binary labels from Python
%  - Build RDHEI path/L_map
%  - Pack header + (labels + dummy msg) bitstream
%  - Embed -> Extract -> Verify label/msg -> Recover carrier (int domain)

clear; clc;

%% ========= [A] CONFIG (EDIT HERE ONLY IF NEEDED) =========
BRIDGE_XYZ = 'E:/bridge/output/demo_case_8192.xyz';
BRIDGE_LAB = 'E:/bridge/output/demo_case_pred_label.bin';

RDHEI_ROOT = 'E:/pointnet-guided-rdhei-medical';
addpath(genpath(RDHEI_ROOT));   % If you ever see weird "SCRIPT false/mfilename" errors, replace with selective addpath.

USE_HUFFMAN = false;            % Recommended OFF for binary labels (already 1 bit/symbol)
MSG_BITS = 512;                 % dummy secret payload to test pipeline
rng(0);
msg = uint8(randi([0,1], 1, MSG_BITS));

cfg = struct();
cfg.nbits  = 64;
cfg.kmax   = 8;
cfg.offset = int64(0);          % will be updated below if needed
%% ========= [A] END CONFIG =========

%% ========= [B] Sanity checks =========
assert(exist(BRIDGE_XYZ,'file')==2, 'Cannot find xyz: %s', BRIDGE_XYZ);
assert(exist(BRIDGE_LAB,'file')==2, 'Cannot find label.bin: %s', BRIDGE_LAB);

%% ========= [C] Load/quantize xyz =========
assert(exist('read_xyz_quantize','file')==2, 'Missing function: read_xyz_quantize.m');
[P_int, meta] = read_xyz_quantize(BRIDGE_XYZ);
assert(size(P_int,2)==3, 'Expected Nx3');
fprintf('Loaded xyz -> P_int size = [%d x %d]\n', size(P_int,1), size(P_int,2));

% Shift to nonnegative int domain if needed (safe default)
minv = min(P_int(:));
if minv < 0
    cfg.offset = int64(-minv);
else
    cfg.offset = int64(0);
end
P0 = int64(P_int) + cfg.offset;
fprintf('Offset used = %d (min before=%d, min after=%d)\n', cfg.offset, int64(minv), min(P0(:)));

%% ========= [D] Load labels (8192 uint8, values 0/1) =========
fid = fopen(BRIDGE_LAB,'rb'); assert(fid>0, 'Cannot open %s', BRIDGE_LAB);
labels_u8 = fread(fid, inf, 'uint8=>uint8'); fclose(fid);

assert(numel(labels_u8)==8192, 'Expected 8192 labels, got %d', numel(labels_u8));
assert(all(labels_u8==0 | labels_u8==1), 'Labels must be 0/1');
fprintf('MATLAB label read OK.\n');

% Ensure carrier point count matches label count
assert(size(P0,1) == numel(labels_u8), 'Point count mismatch: P0=%d labels=%d', size(P0,1), numel(labels_u8));
labels = logical(labels_u8);

%% ========= [E] Build subcloud/path/L_map =========
need1 = {'build_subclouds','choose_startpoint','nn_greedy_path','build_L_map'};
for t = 1:numel(need1)
    assert(exist(need1{t},'file')==2, 'Missing function: %s.m', need1{t});
end

cfg0 = struct(); cfg0.startRule = 'minIndex';
subclouds = build_subclouds(labels);
assert(isstruct(subclouds) && ~isempty(subclouds), 'build_subclouds returned empty');

% Some repos use field "idx", not "i"
if isfield(subclouds(1),'idx')
    idx = subclouds(1).idx;
elseif isfield(subclouds(1),'i')
    idx = subclouds(1).i;
else
    error('subclouds(1) has neither field idx nor i. Inspect build_subclouds output.');
end

start_idx = choose_startpoint(idx, cfg0);
path = nn_greedy_path(P0, idx, start_idx);

pcfg = struct(); pcfg.nbits = cfg.nbits;
[L_map, is_anchor] = build_L_map(P0, path, pcfg);

fprintf('Path length: %d points\n', numel(path));

%% ========= [F] Label bitstream =========
if ~USE_HUFFMAN
    % Binary segmentation labels are already 1 bit per point
    lab_bits = uint8(labels_u8(:).');          % 1x8192
    fprintf('Raw label bits: %d\n', numel(lab_bits));
else
    % Canonical Huffman (advanced; only use if you pack the codebook into header)
    assert(exist('huffman_codelens','file')==2, 'Missing function: huffman_codelens.m');
    assert(exist('canonical_codebook','file')==2, 'Missing function: canonical_codebook.m');
    assert(exist('huff_build_decodemaps','file')==2, 'Missing function: huff_build_decodemaps.m');
    assert(exist('huff_encode_symbols','file')==2, 'Missing function: huff_encode_symbols.m');
    assert(exist('huff_decode_bits','file')==2, 'Missing function: huff_decode_bits.m');

    symbols = labels_u8(:);
    lens = huffman_codelens(symbols);
    cb = canonical_codebook(lens);
    lab_bits = huff_encode_symbols(symbols, cb);
    lab_bits = uint8(lab_bits(:).');
    fprintf('Huffman label bits: %d\n', numel(lab_bits));

    % Self-check (no RDHEI)
    dm0 = huff_build_decodemaps(cb);
    labels_check = huff_decode_bits(lab_bits, dm0, uint32(numel(labels_u8)));
    labels_check = uint8(labels_check(:));
    fprintf('Huff self-check mismatch: %d\n', sum(labels_check ~= labels_u8));
    assert(isequal(labels_check, labels_u8), 'HUFFMAN SELF-CHECK FAILED');
end

%% ========= [G] Header + payload =========
assert(exist('pack_header_bits','file')==2,  'Missing function: pack_header_bits.m');
assert(exist('parse_header_bits','file')==2, 'Missing function: parse_header_bits.m');

H = struct();
H.MAGIC   = hex2dec('5244');
H.VER     = 1;                          % 1=raw labels; 2=huffman labels (reserved)
if USE_HUFFMAN, H.VER = 2; end
H.NBITS   = cfg.nbits;
H.KMAX    = cfg.kmax;
H.OFFSET  = uint64(cfg.offset);
H.N_UNITS  = uint32((numel(path)-1)*3);
H.LAB_BITS = uint32(numel(lab_bits));
H.MSG_BITS = uint32(MSG_BITS);

hb = pack_header_bits(H);
hb = uint8(hb(:).');
bitstream = [hb, lab_bits, msg];
fprintf('Debug: hb_len=%d, lab_len=%d, msg_len=%d\n', numel(hb), numel(lab_bits), numel(msg));
fprintf('Total payload bits = %d\n', numel(bitstream));

%% ========= [H] Embed / Extract / Verify / Recover =========
need2 = {'embed_msbprefix_stream','extract_msbprefix_stream','recover_carrier_msbprefix'};
for t = 1:numel(need2)
    assert(exist(need2{t},'file')==2, 'Missing function: %s.m', need2{t});
end

C0 = P0;
[C_stego, n_written] = embed_msbprefix_stream(C0, C0, path, L_map, is_anchor, bitstream, cfg);
fprintf('Embedded bits: %d / %d\n', n_written, numel(bitstream));
assert(n_written == numel(bitstream), 'Not all bits embedded.');

ext = extract_msbprefix_stream(C_stego, path, L_map, is_anchor, numel(bitstream), cfg);
[H2, used] = parse_header_bits(ext);
fprintf('Parsed VER=%d, LAB_BITS=%d, MSG_BITS=%d\n', H2.VER, H2.LAB_BITS, H2.MSG_BITS);
fprintf('Debug: parse used=%d\n', used);

lab_bits2 = ext(used+1 : used+double(H2.LAB_BITS));
msg2      = ext(used+double(H2.LAB_BITS)+1 : used+double(H2.LAB_BITS)+double(H2.MSG_BITS));

fprintf('Msg bit errors: %d\n', sum(msg2 ~= msg));

% Label bits compare
label_bit_err = sum(lab_bits2 ~= lab_bits);
fprintf('Label bit errors: %d (out of %d)\n', label_bit_err, numel(lab_bits));
if label_bit_err > 0
    k = find(lab_bits2 ~= lab_bits, 1, 'first');
    fprintf('First label-bit mismatch at position k=%d\n', k);
end

% Decode labels
if ~USE_HUFFMAN
    labels_u8_2 = uint8(lab_bits2(:));
else
    dm = huff_build_decodemaps(cb);
    labels_u8_2 = huff_decode_bits(lab_bits2, dm, uint32(numel(labels_u8)));
    labels_u8_2 = uint8(labels_u8_2(:));
end

fprintf('Label mismatch count: %d\n', sum(labels_u8_2 ~= labels_u8));
assert(isequal(labels_u8_2, labels_u8), 'Label decode mismatch!');

% Recover carrier (shifted domain)
C_rec = recover_carrier_msbprefix(C_stego, path, L_map, is_anchor, cfg);
fprintf('Carrier recovery equal to original (shifted int domain): %d\n', isequal(C_rec, C0));
assert(isequal(C_rec, C0), 'Carrier not perfectly recovered!');

disp('E2E PASS: label/msg/carrier all recovered exactly.');
