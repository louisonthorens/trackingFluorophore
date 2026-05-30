function makeTrackingGIF(stack, posTrue, posTrack, filename, opts)
%MAKETRACKINGGIF  Export a single-tracker GIF for the README.
%
%   makeTrackingGIF(STACK, POSTRUE, POSTRACK, FILENAME, OPTS) writes an
%   animated GIF showing the image stack with a fading ground-truth trail
%   (white) and one tracker estimate overlaid each frame (cyan o). All
%   positions are in centred coords (image centre = 0,0).
%
%   Pass [] for posTrue to skip the trail, or [] for posTrack to skip the dot.
%
%   OPTS struct, optional:
%     .trailLen   frames of ground-truth history to show     (30)
%     .clim       display contrast [lo hi]                    (auto from stack)
%     .fps        playback frames per second                  (10)
%     .scale      integer upscaling for a crisper GIF         (8)

    if nargin < 5, opts = struct(); end
    if ~isfield(opts,'trailLen'), opts.trailLen = 30; end
    if ~isfield(opts,'fps'),      opts.fps = 10;      end
    if ~isfield(opts,'scale'),    opts.scale = 8;     end
    if ~isfield(opts,'clim') || isempty(opts.clim)
        lo = double(prctile(stack(:),1));
        hi = double(prctile(stack(:),99.9));
        opts.clim = [lo hi];
    end

    [H, W, nFrames] = size(stack);
    xc = (1:W) - (W+1)/2;          % centred pixel axes
    yc = (1:H) - (H+1)/2;
    delay = 1/opts.fps;

    fig = figure('Color','w','Position',[100 100 60+opts.scale*W 60+opts.scale*H]);
    ax  = axes(fig);

    for t = 1:nFrames
        cla(ax)
        imagesc(ax, xc, yc, double(stack(:,:,t)), opts.clim);
        colormap(ax, gray); axis(ax,'image','off'); hold(ax,'on')

        % fading ground-truth trail
        if ~isempty(posTrue)
            i0 = max(1, t-opts.trailLen); seg = i0:t;
            for k = 1:numel(seg)-1
                a = k/numel(seg) * 0.8;
                p = plot(ax, posTrue(seg(k:k+1),1), posTrue(seg(k:k+1),2), '-');
                p.Color = [0 1 1 a]; p.LineWidth = 0.5 + 2*a;
            end
        end

        % tracker overlay at the current frame (cyan)
        if ~isempty(posTrack)
            plot(ax, posTrack(t,1), posTrack(t,2), 'o', 'Color','red', ...
                 'MarkerSize',11,'LineWidth',2);
        end
        drawnow

        % capture and append to the GIF
        frameImg = frame2im(getframe(ax));
        [idx, map] = rgb2ind(frameImg, 256);
        if t == 1
            imwrite(idx, map, filename, 'gif', 'LoopCount', Inf, 'DelayTime', delay);
        else
            imwrite(idx, map, filename, 'gif', 'WriteMode', 'append', 'DelayTime', delay);
        end
    end
    close(fig)
    fprintf('wrote %s (%d frames)\n', filename, nFrames);
end