function [X, Y] = generateTrainingData(nSamples, cfg)
%GENERATETRAININGDATA  Build labelled patches to train the localisation net.
%
%   [X, Y] = generateTrainingData(NSAMPLES, CFG) generates NSAMPLES training
%   examples by rendering a single particle at a known sub-pixel position,
%   running the SHARED detection front-end, and recording:
%       X : the normalised patch around the detected peak  (P-by-P-by-1-by-N)
%       Y : the label = offset of the TRUE position from the patch centre,
%           in [dx dy] pixels                               (N-by-2)
%
%   WHY THE LABEL IS AN OFFSET, NOT AN ABSOLUTE POSITION
%   A cropped patch looks identical wherever it came from in the full image,
%   so absolute position is not inferable from the patch -- only the
%   particle's position RELATIVE to the crop is. Labelling with the offset
%   from the detected peak is what makes the regression well-posed, and it
%   matches what the tracker adds back at inference.
%
%   Randomising signal and background across the same ranges you will test
%   is what lets the trained net generalise across the SNR sweep rather than
%   memorising one condition.
%
%   CFG  struct, all fields optional (defaults in parentheses):
%     .imgSize        render size [H W] in px                   ([32 32])
%     .patchHalf      patch half-width; patch is 2*half+1 px     (7)
%     .sigmaPSF       PSF std passed to the simulator, px        (1.3)
%     .filterSigma    detection smoothing std, px                (1.3)
%     .posJitter      true position drawn in +/- this, px        (5)
%     .signalRange    [min max] total photons                    ([200 5000])
%     .backgroundRange[min max] background photons/px            ([2 30])
%
%   Requires simulateParticleImage.m and detectAndExtract.m on the path.

    if nargin < 2, cfg = struct(); end
    cfg = setDefault(cfg, 'imgSize',         [32 32]);
    cfg = setDefault(cfg, 'patchHalf',       7);
    cfg = setDefault(cfg, 'sigmaPSF',        1.3);
    cfg = setDefault(cfg, 'filterSigma',     1.3);
    cfg = setDefault(cfg, 'posJitter',       5);
    cfg = setDefault(cfg, 'signalRange',     [200 5000]);
    cfg = setDefault(cfg, 'backgroundRange', [2 30]);

    H = cfg.imgSize(1);  W = cfg.imgSize(2);
    P = 2 * cfg.patchHalf + 1;

    X = zeros(P, P, 1, nSamples, 'single');
    Y = zeros(nSamples, 2, 'single');

    for n = 1:nSamples
        % random true position (centred coords) and imaging conditions
        truePos = (rand(1, 2) * 2 - 1) * cfg.posJitter;

        ip = struct();
        ip.sigmaPSF   = cfg.sigmaPSF;
        % log-uniform signal: spreads samples evenly across orders of
        % magnitude, so low-SNR cases are well represented (linear sampling
        % would crowd most samples at high signal).
        logS          = log10(cfg.signalRange);
        ip.signal     = 10^(logS(1) + rand * diff(logS));
        ip.background = cfg.backgroundRange(1) + rand * diff(cfg.backgroundRange);

        img = simulateParticleImage(truePos, cfg.imgSize, ip);

        % shared front-end: same detection + patch as the tracker uses
        [rx, ry, patch] = detectAndExtract(img, cfg.filterSigma, cfg.patchHalf);

        % detected-peak centre in centred coordinates
        peakC = [rx - (W + 1) / 2, ry - (H + 1) / 2];

        X(:, :, 1, n) = single(patch);
        Y(n, :)       = single(truePos - peakC);   % sub-pixel residual to learn
    end
end

% ------------------------------------------------------------------------
function s = setDefault(s, field, value)
    if ~isfield(s, field) || isempty(s.(field))
        s.(field) = value;
    end
end
