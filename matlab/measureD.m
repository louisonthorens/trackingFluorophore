function D = measureD(pos, params)
%MEASURED  Diffusion coefficient from a 2D trajectory via MSD analysis.
%
%   D = measureD(POS, PARAMS) estimates the diffusion coefficient (um^2/s)
%   of a freely diffusing particle from its trajectory POS (N-by-2, in
%   pixels). It computes the time-averaged mean squared displacement (MSD)
%   over a range of lag times and fits a straight line to the first few lags:
%
%       MSD(tau) = 4*D*tau + 4*sigmaLoc^2          (2D free diffusion)
%
%   The SLOPE gives an unbiased D; the INTERCEPT gives the static
%   localisation precision sigmaLoc. Fitting the slope (rather than forcing
%   the line through the origin) is what removes the localisation-error bias.
%
%   INPUTS
%     POS      N-by-2 trajectory in pixels (any coordinate origin -- only
%              displacements are used).
%     PARAMS   struct, all fields optional (defaults in parentheses):
%        .pixelSize  pixel size in um/px                          (0.1625)
%        .dt         frame interval in s                          (0.1)
%        .maxLag     largest lag (frames) used for the MSD curve  (round(N/4))
%        .nFitLags   number of initial lags used in the line fit  (4)
%
%   OUTPUTS
%     D     diffusion coefficient in um^2/s, from the MSD slope.


    if nargin < 2, params = struct(); end
    N = size(pos, 1);
    params = setDefault(params, 'pixelSize', 0.1625);
    params = setDefault(params, 'dt',        0.1);
    params = setDefault(params, 'maxLag',    max(4, round(N / 4)));
    params = setDefault(params, 'nFitLags',  4);

    maxLag   = min(params.maxLag, N - 1);
    nFitLags = min(params.nFitLags, maxLag);

    % work in physical units; only relative displacements matter
    posUm = pos * params.pixelSize;

    % time-averaged MSD over overlapping windows
    tau = (1:maxLag)' * params.dt;
    msd = zeros(maxLag, 1);
    for n = 1:maxLag
        d = posUm(1+n:end, :) - posUm(1:end-n, :);   % all displacements at lag n
        msd(n) = mean(sum(d.^2, 2));                 % 2D squared displacement
    end

    % linear fit on the first few lags
    k = 1:nFitLags;
    c = polyfit(tau(k), msd(k), 1);                  % c(1)=slope, c(2)=intercept
    slope     = c(1);
    intercept = c(2);

    D        = slope / 4;                            % 2D: MSD = 4*D*tau
end

function s = setDefault(s, field, value)
    if ~isfield(s, field) || isempty(s.(field))
        s.(field) = value;
    end
end