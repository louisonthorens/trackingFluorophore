function net = trainParticleNet(X, Y, opts)
%TRAINPARTICLENET  Train a small CNN to regress a patch -> sub-pixel offset.
%
%   NET = trainParticleNet(X, Y) defines and trains a compact convolutional
%   network that maps a normalised patch (from generateTrainingData) to the
%   2D offset [dx dy] of the particle from the patch centre. The loss is plain
%   mean squared error; the output is two numbers. That is the whole model.
%
%   NET = trainParticleNet(X, Y, OPTS) overrides training options:
%     .maxEpochs    (30)
%     .miniBatch    (256)
%     .learnRate    (1e-3)
%     .valFraction  fraction held out for validation (0.1)
%
%   Requires the Deep Learning Toolbox. This version uses the trainNetwork
%   API with a regressionLayer (works on pre-R2023b MATLAB). On R2023b+ you
%   can instead drop the regressionLayer and call:
%       net = trainnet(Xtr, Ytr, layers, "mse", trainOpts);
%
%   X is P-by-P-by-1-by-N, Y is N-by-2.

    if nargin < 3, opts = struct(); end
    opts = setDefault(opts, 'maxEpochs',   30);
    opts = setDefault(opts, 'miniBatch',   256);
    opts = setDefault(opts, 'learnRate',   1e-3);
    opts = setDefault(opts, 'valFraction', 0.1);

    P = size(X, 1);
    K = size(X, 3);     % channels: 1 for single-frame, >1 for a temporal window
    N = size(X, 4);

    % --- validation split -------------------------------------------------
    idx   = randperm(N);
    nVal  = round(opts.valFraction * N);
    vIdx  = idx(1:nVal);
    tIdx  = idx(nVal+1:end);
    Xtr = X(:, :, :, tIdx);  Ytr = Y(tIdx, :);
    Xval = X(:, :, :, vIdx); Yval = Y(vIdx, :);

    % --- architecture: patch -> [dx dy] ----------------------------------
    layers = [
        imageInputLayer([P P K], 'Normalization', 'none')

        convolution2dLayer(3, 16, 'Padding', 'same')
        reluLayer
        convolution2dLayer(3, 32, 'Padding', 'same')
        reluLayer
        maxPooling2dLayer(2, 'Stride', 2)

        convolution2dLayer(3, 64, 'Padding', 'same')
        reluLayer

        fullyConnectedLayer(64)
        reluLayer
        fullyConnectedLayer(2)             % regression output: dx, dy
        regressionLayer ];                 % carries the MSE loss (trainNetwork API)

    % --- training options -------------------------------------------------
    trainOpts = trainingOptions('adam', ...
        'MaxEpochs',        opts.maxEpochs, ...
        'MiniBatchSize',    opts.miniBatch, ...
        'InitialLearnRate', opts.learnRate, ...
        'Shuffle',          'every-epoch', ...
        'ValidationData',   {Xval, Yval}, ...
        'ValidationFrequency', 50, ...
        'Plots',            'training-progress', ...
        'ExecutionEnvironment', 'auto', ...
        'Verbose',          true);

    % --- train (MSE regression via regressionLayer) ----------------------
    net = trainNetwork(Xtr, Ytr, layers, trainOpts);
end

% ------------------------------------------------------------------------
function s = setDefault(s, field, value)
    if ~isfield(s, field) || isempty(s.(field))
        s.(field) = value;
    end
end
