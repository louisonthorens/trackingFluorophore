close all; clear all; clc

% Generating two GIFs, one at low signal-to-noise ratio (SNR), other with
% high SNR

for sig = [300 1000]
    rng(1)
    pos   = brownianMotion(120, struct('D',20,'dt',1e-3));

    stack = zeros([100 100 120],'uint16');

    ip = struct('signal',sig,'background',100);
    for t = 1:120, stack(:,:,t) = simulateParticleImage(pos(t,:),[100 100],ip); end

    posFit = trackGaussianFit(stack, struct('sigmaPSF',1.3));

    makeTrackingGIF(stack, pos, posFit, sprintf('data/track_signal_ML%d.gif',sig));
end

%% ML-based tracking

close all; clear all; clc

% Generating two GIFs, one at low signal-to-noise ratio (SNR), other with
% high SNR

% First, training the ML with simulated data

cfg = struct('signalRange',[50 500]);
[X, Y] = generateTrainingData(5000, cfg);

% 2. train
net = trainParticleNet(X, Y);
%%

for sig = [300 1000]
    rng(1)
    pos   = brownianMotion(120, struct('D',20,'dt',1e-3));

    stack = zeros([100 100 120],'uint16');

    ip = struct('signal',sig,'background',100);
    for t = 1:120, stack(:,:,t) = simulateParticleImage(pos(t,:),[100 100],ip); end

    posML  = trackMLmatlab(stack, net, struct('patchHalf',7));

    makeTrackingGIF(stack, pos, posML, sprintf('data/track_signal_ML%d.gif',sig));
end