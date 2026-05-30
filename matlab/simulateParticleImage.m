function img = simulateParticleImage(pos, imgSize, params)
%SIMULATEPARTICLEIMAGE  Synthetic SCMOS camera image of fluorescent single
%particle
%
%   img = simulateParticleImage(POS, IMGSIZE, PARAMS) renders one or more
%   diffraction-limited point sources at sub-pixel positions POS on a camera
%   of size IMGSIZE, applies a Gaussian PSF, Poisson shot noise, camera gain
%   and read noise, and returns a UINT image as an sCMOS sensor would.
%
%   The PSF is integrated analytically over each pixel with ERF, so sub-pixel
%   float positions are handled exactly and the PSF sums to 1 (i.e. PARAMS.signal
%   is the total detected-photon budget per particle).
%
%   INPUTS
%     POS      N-by-2 array of particle positions [x y] in pixel units, where
%              x is horizontal (column) and y is vertical (row). The ORIGIN
%              [0 0] is the centre of the image; +x is right, +y is down.
%              Float values OK. For W columns, valid x spans (-W/2, +W/2).
%     IMGSIZE  [H W] image size in pixels.
%     PARAMS   struct, all fields optional (defaults in parentheses):
%        .signal      total photons per particle, scalar or N-by-1    (500)
%        .sigmaPSF    PSF std in pixels, scalar or [sigmaX sigmaY]    (1.3)
%        .background  background photons per pixel (has shot noise)   (5)
%        .QE          quantum efficiency, photon -> electron          (0.95)
%        .gain        sensor gain, electrons per ADU                  (0.64)
%        .readNoise   read-noise std in electrons                     (1.0)
%        .offset      constant camera black level ADU                 (119)
%        .bitDepth    bit depth; output clipped to 2^bitDepth-1       (16)
%
%   OUTPUTS
%     img        IMGSIZE UINT image as recorded by the camera.
%
%   EXAMPLE
%     p.signal = 2000; p.background = 8;
%     [img, ph] = simulateParticleImage([0.4 0.7], [32 32], p);  % near centre
%     imagesc(img); axis image; colormap gray
%
%   Defaults values are matching typical microscope camera specs

    % defaults values
    if nargin < 3, params = struct(); end
    params = setDefault(params, 'signal',     500);
    params = setDefault(params, 'sigmaPSF',   1.3);
    params = setDefault(params, 'background', 5);
    params = setDefault(params, 'QE',         0.95);
    params = setDefault(params, 'gain',       0.64);
    params = setDefault(params, 'readNoise',  1.0);
    params = setDefault(params, 'offset',     100);
    params = setDefault(params, 'bitDepth',   16);

    H = imgSize(1);  W = imgSize(2);
    N = size(pos, 1);

    % allow anisotropic PSF
    sig = params.sigmaPSF(:).';
    if isscalar(sig), sig = [sig sig]; end
    sx = sig(1);  sy = sig(2);

    % allow per-particle signal, scalar broadcasts to all
    sig0 = params.signal(:);
    if isscalar(sig0), sig0 = repmat(sig0, N, 1); end

    % render expected photons via pixel-integrated Gaussian PSF
    % Pixel grid centred on the image: position [0 0] is the image centre.
    % Pixel centres run from -(W-1)/2 to +(W-1)/2; edges are shifted likewise.
    xEdges = (0.5 : 1 : W + 0.5) - (W + 1)/2;   % 1-by-(W+1), centred on 0
    yEdges = (0.5 : 1 : H + 0.5) - (H + 1)/2;   % 1-by-(H+1), centred on 0

    photonImg = zeros(H, W);
    for k = 1:N
        x0 = pos(k, 1);  y0 = pos(k, 2);
        % fraction of the unit-volume Gaussian falling in each pixel column/row
        colFrac = diff(0.5 * erf((xEdges - x0) ./ (sqrt(2) * sx)));   % 1-by-W
        rowFrac = diff(0.5 * erf((yEdges - y0) ./ (sqrt(2) * sy)));   % 1-by-H
        photonImg = photonImg + sig0(k) * (rowFrac.' * colFrac);     % H-by-W
    end
    photonImg = photonImg + params.background;   % uniform background photons

    % camera noise model
    % 1. shot noise on detected photoelectrons (combines emission + detection)
    elec = poissrnd(params.QE * photonImg);
    % 2. read noise, Gaussian in electrons
    elec = elec + params.readNoise * randn(H, W);
    % 3. convert electrons -> ADU and add the constant black signal
    adu  = elec ./ params.gain + params.offset;
    % 4. quantise and clip to the ADC range
    adu  = round(adu);
    adu  = min(max(adu, 0), 2^params.bitDepth - 1);

    img = uint16(adu);
end


function s = setDefault(s, field, value)
    if ~isfield(s, field) || isempty(s.(field))
        s.(field) = value;
    end
end