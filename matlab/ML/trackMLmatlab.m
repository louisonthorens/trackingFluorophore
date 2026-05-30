function [pos, info] = trackMLmatlab(stack, net, params)
%TRACKMLMATLAB  ML single-particle tracker (learned sub-pixel refinement).
%
%   POS = trackMLmatlab(STACK, NET, PARAMS) localises one particle per frame
%   using the SAME detection front-end as the classical tracker, then refines
%   the sub-pixel position with a trained network NET instead of a Gaussian
%   fit. Returned POS uses the centred convention ([0 0] = image centre), so
%   it is directly comparable to the ground truth and to trackGaussianFit.
%
%   The per-frame logic mirrors trackGaussianFit exactly up to the refinement:
%       detect peak (shared)  ->  crop+normalise patch (shared)  ->  net offset
%   so any accuracy difference is attributable to fit-vs-learned, nothing else.
%
%   PARAMS struct, optional:
%     .patchHalf    must match the value used in training        (7)
%     .filterSigma  detection smoothing std, px                  (1.3)
%
%   OUTPUTS
%     POS    NFRAMES-by-2 positions [x y] in centred pixel coords.
%     info   struct with .offset (NFRAMES-by-2 predicted sub-pixel offsets).
%
%   Requires detectAndExtract.m on the path and a NET from trainParticleNet.

    if nargin < 3, params = struct(); end
    params = setDefault(params, 'patchHalf',   7);
    params = setDefault(params, 'filterSigma', 1.3);

    [H, W, nFrames] = size(stack);

    pos    = nan(nFrames, 2);
    offset = nan(nFrames, 2);

    for t = 1:nFrames
        frame = double(stack(:, :, t));

        % shared detection + patch (identical to training)
        [rx, ry, patch] = detectAndExtract(frame, params.filterSigma, params.patchHalf);

        % learned sub-pixel offset; force a 1-by-2 row regardless of layout.
        % Works for both a dlnetwork (from trainParticleNetOTF) and a
        % SeriesNetwork/DAGNetwork (from trainParticleNet/trainNetwork).
        P = 2 * params.patchHalf + 1;
        xin = reshape(single(patch), [P P 1 1]);
        if isa(net, 'dlnetwork')
            d = extractdata(predict(net, dlarray(xin, 'SSCB')));
        else
            d = predict(net, xin);
        end
        d = double(d(:)).';

        % detected-peak centre in centred coords, plus the learned offset
        peakC = [rx - (W + 1) / 2, ry - (H + 1) / 2];
        pos(t, :)    = peakC + d;
        offset(t, :) = d;
    end

    info = struct('offset', offset);
end

% ------------------------------------------------------------------------
function s = setDefault(s, field, value)
    if ~isfield(s, field) || isempty(s.(field))
        s.(field) = value;
    end
end
