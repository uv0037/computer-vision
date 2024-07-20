%%

function image_feats = get_colour_histograms(image_paths, numBins, color_space)
    N = length(image_paths);
    d = 3 * numBins;  
    image_feats = zeros(N, d);
    
    for i = 1:N
        img = imread(image_paths{i});
        if strcmp(color_space, 'RGB')
            img = img;
        elseif strcmp(color_space, 'HSV')
            img = rgb2hsv(img);
        elseif strcmp(color_space, 'YCbCr')
            img = rgb2ycbcr(img);
        elseif strcmp(color_space, 'lab')
            img = rgb2lab(img);
        else
            error('Unsupported color space');
        end
        R = img(:,:,1);
        G = img(:,:,2);
        B = img(:,:,3);
        
        [rHist, ~] = histcounts(R, numBins);
        [gHist, ~] = histcounts(G, numBins);
        [bHist, ~] = histcounts(B, numBins);
        

        rHist = rHist / sum(rHist);
        gHist = gHist / sum(gHist);
        bHist = bHist / sum(bHist);
        

        image_feats(i, :) = [rHist gHist bHist];
    end
end