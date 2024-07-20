%%

function image_feats = get_tiny_images(image_paths)

    tinySize = [4 4];  % 2X2,4X4,16x16,256X256 pixels
    

    N = length(image_paths);
    d = prod(tinySize);  
    image_feats = zeros(N, d);
    
    for i = 1:N
        img = imread(image_paths{i});
        
        if size(img, 3) == 3
            %img = rgb2gray(img);
            %img = rgb2ind(img, 5)
            img = rgb2hsv(img);
        end
        

        tinyImg = imresize(img, tinySize, 'bilinear');
        

        tinyImgVector = double(tinyImg(:));  
        tinyImgVector = tinyImgVector - mean(tinyImgVector); 
        tinyImgVector = tinyImgVector / std(tinyImgVector);  
        

        image_feats(i, :) = tinyImgVector(1:16)';
    end
end