function start_idx = choose_startpoint(idx, cfg)
%CHOOSE_STARTPOINT Choose a deterministic start point within a subcloud.
% idx: vector of point indices (1-based)
% cfg.startRule: 'minIndex' or 'keyed'

    if isempty(idx)
        error('Empty subcloud index list.');
    end

    if ~isfield(cfg, 'startRule')
        cfg.startRule = 'minIndex';
    end

    switch cfg.startRule
        case 'minIndex'
            start_idx = min(idx);

        case 'keyed'
            % Simple keyed rule (deterministic): use sum(idx) + key to pick one
            if ~isfield(cfg, 'KD')
                error('cfg.KD is required for startRule = keyed');
            end
            key = double(sum(uint8(cfg.KD))); % cfg.KD can be a string
            pos = mod(sum(double(idx)) + key, numel(idx)) + 1;
            start_idx = idx(pos);

        otherwise
            error('Unknown startRule: %s', cfg.startRule);
    end
end
