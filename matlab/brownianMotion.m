function [pos, stepStd_px] = brownianMotion(nFrames, params)
%BROWNIANMOTION  2D free-diffusion trajectory in pixel units.
%
%   POS = brownianMotion(NFRAMES, PARAMS) returns an NFRAMES-by-2 array of
%   [x y] particle positions (in pixels) for a freely diffusing particle.
%
%   Physics: each frame the particle takes an independent Gaussian step with
%   per-axis variance 2*D*dt. D and dt are given in
%   physical units and converted to pixels via the pixel size.
%
%   INPUTS
%     NFRAMES  number of time points (frames) to generate.
%     PARAMS   struct, all fields optional (defaults in parentheses):
%        .D          diffusion coefficient in um^2/s              (0.1)
%        .dt         frame interval in s (1/frame rate)           (0.1)
%        .pixelSize  pixel size in um/px                          (0.1625)
%        .start      starting position [x0 y0] in px              ([0 0])
%
%   OUTPUTS
%     POS         NFRAMES-by-2 trajectory in pixels.
%     stepStd_px  per-axis step standard deviation in pixels
%                 ( = sqrt(2*D*dt)/pixelSize ), handy for bookkeeping.
%
%   EXAMPLE
%     p.D = 0.1; p.dt = 0.1;
%     pos = brownianMotion(200, p);
%     plot(pos(:,1), pos(:,2), '-o'); axis equal

    if nargin < 2, params = struct(); end
    params = setDefault(params, 'D',         0.1);
    params = setDefault(params, 'dt',        0.1);
    params = setDefault(params, 'pixelSize', 0.1625);
    params = setDefault(params, 'start',     [0 0]);

    % per-axis step size: physical std sqrt(2*D*dt), converted to pixels
    stepStd_px = sqrt(2 * params.D * params.dt) / params.pixelSize;

    % first frame at the start position, then accumulate Gaussian steps
    steps        = stepStd_px * randn(nFrames, 2);
    steps(1, :)  = 0;                              % no step into frame 1
    pos          = params.start + cumsum(steps, 1);
end

function s = setDefault(s, field, value)
    if ~isfield(s, field) || isempty(s.(field))
        s.(field) = value;
    end
end