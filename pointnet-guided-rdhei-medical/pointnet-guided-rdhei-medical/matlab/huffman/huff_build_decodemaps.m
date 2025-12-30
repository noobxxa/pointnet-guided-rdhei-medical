function dm = huff_build_decodemaps(cb)
%HUFF_BUILD_DECODEMAPS Build decode maps per code length for fast decoding.
% dm.lenList{L} : vector of codes (uint32) at length L
% dm.symList{L} : corresponding symbols (uint16) at length L
% dm.maxLen

    S = numel(cb.lens);
    maxLen = cb.maxLen;

    dm = struct();
    dm.maxLen = maxLen;
    dm.lenList = cell(maxLen,1);
    dm.symList = cell(maxLen,1);

    for L = 1:maxLen
        dm.lenList{L} = uint32([]);
        dm.symList{L} = uint16([]);
    end

    for s = 1:S
        L = double(cb.lens(s));
        if L > 0
            dm.lenList{L}(end+1,1) = uint32(cb.code(s)); %#ok<AGROW>
            dm.symList{L}(end+1,1) = uint16(s-1);        %#ok<AGROW>
        end
    end
end
