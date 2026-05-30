function [pos, fitInfo] = trackGaussianFit(stack, params)
%TRACKGAUSSIANFIT  Classical single-particle localisation by 2D Gaussian fit.
%
%   POS = trackGaussianFit(STACK, PARAMS) localises one bright particle in
%   every frame of an image STACK. For each frame it (1) smooths with a small
%   Gaussian to suppress noise, (2) finds the brightest pixel as a coarse
%   guess, (3) least-squares fits a 2D Gaussian on a small ROI around it, and
%   returns the sub-pixel centre. With a single particle per frame this IS the
%   classical track (no linking step needed).
%
%   Coordinates match simulateParticleImage / brownianMotion: the returned
%   POS uses [0 0] = image centre, +x right, +y down, so it is directly
%   comparable to the ground-truth trajectory.
%
%   Uses only base MATLAB (conv2 + fminsearch) -- no toolboxes required.
%
%   INPUTS
%     STACK    H-by-W-by-NFRAMES image stack (uint16 or double).
%     PARAMS   struct, all fields optional (defaults in parentheses):
%        .sigmaPSF      expected PSF std in px, used as fit init   (1.3)
%        .filterSigma   Gaussian smoothing std for detection, px   (sigmaPSF)
%        .roiHalfWidth  half-size of the square fit ROI, px        (ceil(3*sigmaPSF))
%
%   OUTPUTS
%     POS       NFRAMES-by-2 fitted positions [x y] in centred pixel coords.
%     fitInfo   struct with NFRAMES-by-1 fields for quality filtering:
%        .amplitude   fitted peak amplitude (ADU above background)
%        .sigma       fitted PSF std (px)
%        .background  fitted local background (ADU)
%        .residual    RMS fit residual (ADU) -- high = bad localisation
%
%   EXAMPLE
%     pos = trackGaussianFit(stack, struct('sigmaPSF', 1.3));
%     plot(pos(:,1), pos(:,2), '-o'); axis equal


    if nargin < 2, params = struct(); end
    params = setDefault(params, 'sigmaPSF',     1.3);
    params = setDefault(params, 'filterSigma',  params.sigmaPSF);
    params = setDefault(params, 'roiHalfWidth', ceil(3 * params.sigmaPSF));

    [H, W, nFrames] = size(stack);
    hw  = params.roiHalfWidth;

    % small Gaussian kernel for detection smoothing
    kr  = ceil(3 * params.filterSigma);
    [kx, ky] = meshgrid(-kr:kr, -kr:kr);
    kernel = exp(-(kx.^2 + ky.^2) / (2 * params.filterSigma^2));
    kernel = kernel / sum(kernel(:));

    opts = optimset('Display', 'off', 'TolX', 1e-4, ...
                    'TolFun', 1e-4, 'MaxFunEvals', 2000, 'MaxIter', 2000);

    pos       = nan(nFrames, 2);
    amplitude = nan(nFrames, 1);
    sigmaOut  = nan(nFrames, 1);
    background = nan(nFrames, 1);
    residual  = nan(nFrames, 1);

    for t = 1:nFrames
        frame = double(stack(:, :, t));

        % detect: smooth then take the brightest pixel
        sm = conv2(frame, kernel, 'same');
        [~, idx] = max(sm(:));
        [ry, rx] = ind2sub([H, W], idx);          % integer peak (row, col)

        % extract fit ROI, clamped to image edges
        x1 = max(1, rx - hw);  x2 = min(W, rx + hw);
        y1 = max(1, ry - hw);  y2 = min(H, ry + hw);
        roi = frame(y1:y2, x1:x2);
        [xx, yy] = meshgrid(x1:x2, y1:y2);        % absolute pixel coords

        % initial guess
        B0 = median(roi(:));
        A0 = max(roi(:)) - B0;
        p0 = [A0, rx, ry, params.sigmaPSF, B0];   % [amp x0 y0 sigma bg]

        % 2D Gaussian least-squares fit
        % sigma enters squared so its sign is irrelevant; abs() applied after.
        model = @(p) p(1) * exp(-((xx - p(2)).^2 + (yy - p(3)).^2) ...
                                 ./ (2 * p(4)^2)) + p(5);
        obj   = @(p) sum((model(p) - roi).^2, 'all');
        pfit  = fminsearch(obj, p0, opts);

        % store (convert pixel index -> centred coordinates)
        pos(t, 1)    = pfit(2) - (W + 1) / 2;
        pos(t, 2)    = pfit(3) - (H + 1) / 2;
        amplitude(t) = pfit(1);
        sigmaOut(t)  = abs(pfit(4));
        background(t) = pfit(5);
        residual(t)  = sqrt(obj(pfit) / numel(roi));
    end

    fitInfo = struct('amplitude', amplitude, 'sigma', sigmaOut, ...
                     'background', background, 'residual', residual);
end

function s = setDefault(s, field, value)
    if ~isfield(s, field) || isempty(s.(field))
        s.(field) = value;
    end
end