function [P_int, meta] = read_xyz_quantize(filepath)
%READ_XYZ_QUANTIZE
% Read xyz-like files with >=3 columns (e.g., x y z [nx ny nz ...]),
% quantize floating-point coordinates to int64 by decimal scaling.
%
% Compatible with older MATLAB versions (no "contains").

    fid = fopen(filepath, 'r');
    if fid < 0
        error('Cannot open file: %s', filepath);
    end

    % Read whole file line-by-line (keeps extra columns safe)
    lines = textscan(fid, '%s', 'Delimiter', '\n', 'Whitespace', '', 'CommentStyle', '#');
    fclose(fid);

    lines = lines{1};
    if isempty(lines)
        error('Empty or invalid xyz file: %s', filepath);
    end

    N = numel(lines);
    sx = cell(N,1);
    sy = cell(N,1);
    sz = cell(N,1);

    for i = 1:N
        tline = strtrim(lines{i});
        if isempty(tline)
            error('Empty line encountered at line %d.', i);
        end

        tokens = strsplit_compat(tline); % compatible splitter
        if numel(tokens) < 3
            error('Line %d has fewer than 3 columns.', i);
        end

        sx{i} = tokens{1};
        sy{i} = tokens{2};
        sz{i} = tokens{3};
    end

    % Determine maximum decimal digits among x,y,z
    dx = max_decimal_digits(sx);
    dy = max_decimal_digits(sy);
    dz = max_decimal_digits(sz);

    d = max([dx, dy, dz]);        % global decimal precision
    S = int64(10)^int64(d);       % scaling factor

    % Convert to double, then quantize
    x = str2double(sx);
    y = str2double(sy);
    z = str2double(sz);

    if any(isnan(x)) || any(isnan(y)) || any(isnan(z))
        error('NaN encountered when converting coordinates. Check file formatting.');
    end

    P_int = int64(round([x, y, z] * double(S)));

    meta = struct();
    meta.decimals = d;
    meta.scale = S;
end

function d = max_decimal_digits(scol)
%MAX_DECIMAL_DIGITS
% Return maximum number of digits after decimal point in a string column.
% Old MATLAB compatible: no "contains".

    d = 0;
    for i = 1:numel(scol)
        s = strtrim(scol{i});
        if isempty(s)
            continue;
        end

        % Scientific notation check: if 'e' or 'E' appears
        s_low = lower(s);
        if ~isempty(strfind(s_low, 'e'))
            % Conservative fallback: assume up to 6 decimals
            d = max(d, 6);
            continue;
        end

        k = strfind(s, '.');
        if isempty(k)
            di = 0;
        else
            di = length(s) - k(end);
        end
        d = max(d, di);
    end
end

function tokens = strsplit_compat(s)
%STRSPLIT_COMPAT
% Split by whitespace (space/tab), compatible with older MATLAB (no strsplit needed).
% Returns a cell array of tokens.

    % regexp with \s+ splits on any whitespace
    tokens = regexp(s, '\s+', 'split');
end
