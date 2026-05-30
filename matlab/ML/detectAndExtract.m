function [rx, ry, patch] = detectAndExtract(frame, filterSigma, patchHalf)
%DETECTANDEXTRACT  Shared front-end for the classical and ML trackers.
%
%   [RX, RY, PATCH] = detectAndExtract(FRAME, FILTERSIGMA, PATCHHALF) smooths
%   FRAME with a Gaussian, takes the brightest pixel as a coarse detection
%   (RX, RY in pixel indices), then crops and normalises a square patch of
%   side 2*PATCHHALF+1 centred on that pixel.
%
%   This is called identically during training-data generation and at
%   inference, so the network always sees the same kind of input. The only
%   thing that differs between the classical and ML trackers is what happens
%   AFTER this step (Gaussian fit vs network regression).
%
%   Normalisation removes the background level and the absolute brightness
%   (subtract min, divide by max), so the network keys on the spot SHAPE,
%   not on signal level -- important for generalising across the SNR sweep.

    frame = double(frame);
    [H, W] = size(frame);

    % --- smooth then detect brightest pixel ------------------------------
    kr = ceil(3 * filterSigma);
    [kx, ky] = meshgrid(-kr:kr, -kr:kr);
    kernel = exp(-(kx.^2 + ky.^2) / (2 * filterSigma^2));
    kernel = kernel / sum(kernel(:));

    sm = conv2(frame, kernel, 'same');
    [~, idx] = max(sm(:));
    [ry, rx] = ind2sub([H, W], idx);

    % --- crop a fixed-size patch, clamping indices at the image edges ----
    rows = min(max(ry - patchHalf : ry + patchHalf, 1), H);
    cols = min(max(rx - patchHalf : rx + patchHalf, 1), W);
    patch = frame(rows, cols);

    % --- normalise: drop background offset and brightness ----------------
    patch = patch - min(patch(:));
    m = max(patch(:));
    if m > 0, patch = patch / m; end
end
