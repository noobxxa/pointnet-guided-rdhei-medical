function labels = read_label(filepath)
%READ_LABEL
% Read one integer label per line (old MATLAB compatible).
% Supports .label / .txt files with one number per line.

    fid = fopen(filepath, 'r');
    if fid < 0
        error('Cannot open label file: %s', filepath);
    end

    % Read as strings first (safer for old MATLAB)
    C = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '');
    fclose(fid);

    lines = C{1};
    if isempty(lines)
        error('Empty label file: %s', filepath);
    end

    N = numel(lines);
    labels = zeros(N,1,'int64');

    for i = 1:N
        s = strtrim(lines{i});
        if isempty(s)
            error('Empty line in label file at line %d.', i);
        end

        v = str2double(s);
        if isnan(v)
            error('Non-numeric label at line %d: %s', i, s);
        end

        labels(i) = int64(v);
    end
end
