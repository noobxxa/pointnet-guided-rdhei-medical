% matlab_read_label_example.m
fid = fopen('outputs/preds/demo_case_pred_label.bin','rb');
y = fread(fid, inf, 'uint8=>uint8');
fclose(fid);

assert(numel(y) == 8192, sprintf('Expected 8192 labels, got %d', numel(y)));
y = logical(y);   % N×1 label (0/1)

disp(['Loaded labels: ', num2str(numel(y)), ' points']);
