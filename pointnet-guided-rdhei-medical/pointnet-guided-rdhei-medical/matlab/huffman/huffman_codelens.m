function lens = huffman_codelens(freq)
%HUFFMAN_CODELENS Generate Huffman code lengths from symbol frequencies.
% freq: (S x 1) nonnegative integers/doubles
% lens: (S x 1) uint8 code lengths; 0 means unused symbol
%
% Old MATLAB compatible, no toolboxes.

    freq = double(freq(:));
    S = numel(freq);
    lens = zeros(S,1,'uint8');

    used = find(freq > 0);
    if isempty(used)
        return;
    end
    if numel(used) == 1
        % Only one symbol: assign length 1
        lens(used(1)) = uint8(1);
        return;
    end

    % Build a Huffman tree using a simple O(S^2) merging (S is tiny here).
    % Nodes: 1..S are leaves, S+1.. are internal.
    maxNodes = 2*S;
    parent = zeros(maxNodes,1);
    weight = zeros(maxNodes,1);
    active = zeros(maxNodes,1);

    for i = 1:S
        weight(i) = freq(i);
        if freq(i) > 0
            active(i) = 1;
        end
    end

    nextNode = S + 1;

    while true
        idx = find(active(1:nextNode-1) == 1);
        if numel(idx) <= 1
            break;
        end

        % pick two smallest weights
        w = weight(idx);
        [~, ord] = sort(w, 'ascend');
        a = idx(ord(1));
        b = idx(ord(2));

        % merge
        weight(nextNode) = weight(a) + weight(b);
        parent(a) = nextNode;
        parent(b) = nextNode;

        active(a) = 0;
        active(b) = 0;
        active(nextNode) = 1;

        nextNode = nextNode + 1;
    end

    % compute lengths for leaves
    for i = 1:S
        if freq(i) <= 0
            continue;
        end
        len = 0;
        j = i;
        while parent(j) ~= 0
            len = len + 1;
            j = parent(j);
        end
        lens(i) = uint8(len);
    end

    % Optional: bound lengths (not needed for S<=9 typically)
end
