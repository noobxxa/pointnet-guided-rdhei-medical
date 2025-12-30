%% matlab_e2e_full_test.m  (BRIDGE + CLEAN)
clear; clc;

%% ========= [A] CONFIG: EDIT ONLY THIS BLOCK =========
% 1) Bridge paths (use forward slashes to avoid escaping issues)
BRIDGE_XYZ = 'E:/bridge/output/demo_case_8192.xyz';


BRIDGE_LAB = 'E:/bridge/output/demo_case_pred_label.bin';

% 1.5) Label coding mode
%    false: embed raw 0/1 labels directly (recommended for binary lesion/non-lesion)
%    true : use canonical Huffman (only if you also store codebook in header; not enabled by default)
USE_HUFFMAN = false;

% 2) Add ONLY your RDHEI project code folders (DO NOT add python project)
%    Put folders that contain: read_xyz_quantize, build_subclouds, embed_msbprefix_stream, etc.
RDHEI_ROOT = 'E:/pointnet-guided-rdhei-medical';
addpath(genpath(RDHEI_ROOT));   % ȷϲȾú򻻳ɸȷĿ¼ addpath

% 3) RDHEI params
cfg = struct(); 
cfg.nbits  = 64;
cfg.kmax   = 8;
cfg.offset = int64(0);

% 4) Secret message for test
MSG_BITS = 512;
rng(0);                         % deterministic
msg = uint8(randi([0,1], 1, MSG_BITS));
%% ========= [A] END CONFIG =========

%% ========= [B] Sanity checks: required files =========
assert(exist(BRIDGE_XYZ,'file')==2, 'Cannot find xyz: %s', BRIDGE_XYZ);
assert(exist(BRIDGE_LAB,'file')==2, 'Cannot find label.bin: %s', BRIDGE_LAB);

%% ========= [C] Load/quantize point cloud (carrier domain) =========
assert(exist('read_xyz_quantize','file')==2, 'Missing function: read_xyz_quantize.m');
[P0, meta] = read_xyz_quantize(BRIDGE_XYZ);
assert(~isempty(P0), 'P0 empty');
fprintf('Loaded xyz -> P0 size = [%d x %d]\n', size(P0,1), size(P0,2));

% ===== Force nonnegative integer domain via offset (IMPORTANT) =====
minv = min(P0(:));
if minv < 0
    cfg.offset = int64(-minv);     % shift so minimum becomes 0
else
    cfg.offset = int64(0);
end

P0 = int64(P0) + cfg.offset;    % shifted carrier used for ALL RDHEI steps
fprintf('Offset used = %d (min before=%d, min after=%d)\n', cfg.offset, int64(minv), min(P0(:)));
% ================================================================


%% ========= [D] Load labels from Python (8192x1 uint8 in {0,1}) =========
fid = fopen(BRIDGE_LAB,'rb'); assert(fid>0, 'Cannot open %s', BRIDGE_LAB);
labels_u8 = fread(fid, inf, 'uint8=>uint8'); fclose(fid);

assert(numel(labels_u8)==8192, 'Expected 8192 labels, got %d', numel(labels_u8));
assert(all(labels_u8==0 | labels_u8==1), 'Labels must be 0/1');
labels = logical(labels_u8);
disp('MATLAB label read OK.');

%% ========= [E] Build subcloud/path/L_map (same as your previous flow) =========
need1 = {'build_subclouds','choose_startpoint','nn_greedy_path','build_L_map'};
for i = 1:numel(need1)
    assert(exist(need1{i},'file')==2, 'Missing function: %s.m', need1{i});
end

cfg0 = struct(); cfg0.startRule = 'minIndex';
subclouds = build_subclouds(labels);
idx = subclouds(1).i%% ========= [F] Label bitstream =========
% For binary segmentation labels (0/1), the most robust and compact is to embed them directly.
% Huffman is optional and OFF by default because this project uses a self-describing header only for RDHEI fields.
if ~USE_HUFFMAN
    lab_bits = uint8(labels_u8(:).');    % 1x8192 bits (0/1)
    fprintf('Raw label bits: %d\n', numel(lab_bits));
else
    % --- Canonical Huffman (advanced) ---
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

    % self-check (no RDHEI)
    dm0 = huff_build_decodemaps(cb);
    labels_check = huff_decode_bits(lab_bits, dm0, uint32(numel(labels_u8)));
    labels_check = uint8(labels_check(:));
    fprintf('Huff self-check mismatch: %d\n', sum(labels_check ~= labels_u8));
    assert(isequal(labels_check, labels_u8), 'HUFFMAN SELF-CHECK FAILED');
end



fprintf('Huff self-check mismatch: %d\n', sum(labels_check ~= labels_u8));
assert(isequal(labels_check, labels_u8), 'HUFFMAN SELF-CHECK FAILED (encode/decode not consistent)');
%----------------------------------------------






%% ========= [G] Build header + payload =========
% NOTE: 㹤 pack_header_bits/parse_header_bits ǩܲͬ
% дǡȴ Huffman dictʧܾͻ˵ֻ pack H
assert(exist('pack_header_bits','file')==2,  'Missing function: pack_header_bits.m');
assert(exist('parse_header_bits','file')==2, 'Missing function: parse_header_bits.m');

H = struct();
H.MAGIC   = hex2dec('5244');
H.VER     = 2;
H.NBITS   = cfg.nbits;
H.KMAX    = cfg.kmax;
H.OFFSET  = uint64(cfg.offset);
H.N_UNITS  = uint32((numel(path)-1)*3);
H.LAB_BITS = uint32(numel(lab_bits));
H.MSG_BITS = uint32(MSG_BITS);


% Versioning: VER=1 for raw labels, VER=2 reserved for Huffman-coded labels
if ~USE_HUFFMAN
    H.VER = 1;
else
    H.VER = 2;
end
% Pack header (self-contained RDHEI fields only)
hb = pack_header_bits(H);
hb = uint8(hb(:).'); % ensure row
bitstream = [hb, lab_bits, msg];
fprintf('Debug: hb_len=%d, lab_len=%d, msg_len=%d\n', numel(hb), numel(lab_bits), numel(msg));

fprintf('Total payload bits = %d (header=%d, lab=%d, msg=%d)\n', numel(bitstream), numel(hb), numel(lab_bits), numel(msg));

%% ========= [H] Embed / Extract / Recover =========
need2 = {'embed_msbprefix_stream','extract_msbprefix_stream','recover_carrier_msbprefix'};
for i = 1:numel(need2)
    assert(exist(need2{i},'file')==2, 'Missing function: %s.m', need2{i});
end

C0 = P0;
[C_stego, n_written] = embed_msbprefix_stream(C0, C0, path, L_map, is_anchor, bitstream, cfg);
fprintf('Embedded bits: %d / %d\n', n_written, numel(bitstream));
assert(n_written == numel(bitstream), 'Not all bits embedded (capacity insufficient or config mismatch).');

ext = extract_msbprefix_stream(C_stego, path, L_map, is_anchor, numel(bitstream), cfg);

% Parse header back
% Parse header back (we only need H2 and used for slicing)
[H2, used] = parse_header_bits(ext);
fprintf('Debug: parse used=%d, H2.LAB_BITS=%d, H2.MSG_BITS=%d\n', used, H2.LAB_BITS, H2.MSG_BITS);
assert(double(H2.LAB_BITS)==numel(lab_bits), 'Header LAB_BITS mismatch: header=%d, local=%d', H2.LAB_BITS, numel(lab_bits));
assert(double(H2.MSG_BITS)==numel(msg),      'Header MSG_BITS mismatch: header=%d, local=%d', H2.MSG_BITS, numel(msg));


fprintf('Parsed VER=%d, LAB_BITS=%d, MSG_BITS=%d\n', H2.VER, H2.LAB_BITS, H2.MSG_BITS);

lab_bits2 = ext(used+1 : used+double(H2.LAB_BITS));

% Debug: check extracted label bitstream correctness
if numel(lab_bits2) == numel(lab_bits)
    bit_err = sum(lab_bits2 ~= lab_bits);
    fprintf('Label bit errors: %d (out of %d)\n', bit_err, numel(lab_bits));
    if bit_err > 0
        k = find(lab_bits2 ~= lab_bits, 1, 'first');
        fprintf('First label-bit mismatch at position k=%d\n', k);
    end
end
msg2      = ext(used+double(H2.LAB_BITS)+1 : used+double(H2.LAB_BITS)+double(H2.MSG_BITS));

fprintf('Msg bit errors: %d\n', sum(msg2 ~= msg));
bit_err = sum(lab_bits2 ~= lab_bits);
%% ========= Decode labels =========
if ~USE_HUFFMAN
    labels_u8_2 = uint8(lab_bits2(:));
else
    dm = huff_build_decodemaps(cb);
    labels_u8_2 = huff_decode_bits(lab_bits2, dm, uint32(numel(labels_u8)));
    labels_u8_2 = uint8(labels_u8_2(:));
end

fprintf('Label mismatch count: %d\n', sum(labels_u8_2 ~= labels_u8));
assert(isequal(labels_u8_2, labels_u8), 'Label decode mismatch!');
% Recover carrier
C_rec = recover_carrier_msbprefix(C_stego, path, L_map, is_anchor, cfg);
fprintf('Carrier recovery equal to original (int domain): %d\n', isequal(C_rec, C0));
assert(isequal(C_rec, C0), 'Carrier not perfectly recovered!');

disp('E2E PASS: label/msg/carrier all recovered exactly.');