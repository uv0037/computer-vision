function [train_image_feats, test_image_feats] = get_spatial_pyramids(train_image_paths, test_image_paths, vocab_size, max_level)
    vocab_path = ['vocab_size_spatial_gray',num2str(vocab_size),'.mat'];
    if exist(vocab_path, 'file')
        load(vocab_path, 'vocab');
    else
        all_features = extract_features(train_image_paths, vocab_size, 'color', max_level);
        vocab = vl_kmeans(double(all_features), vocab_size)
        save(vocab_path, 'vocab');
    end


    train_image_feats = extract_spatial_pyramid_feats(train_image_paths, vocab, vocab_size, max_level);
    test_image_feats = extract_spatial_pyramid_feats(test_image_paths, vocab, vocab_size, max_level);
end

function all_features = extract_features(image_paths, vocab_size, type, max_level)
    all_features = [];
    for i = 1:length(image_paths)
        img = single(imread(image_paths{i}));
        if strcmp(type, 'color')
            for channel = 1:3
                img_channel = img(:, :, channel);
                all_features = [all_features, get_features(img_channel, max_level)];
            end
        elseif strcmp(type, 'gray')
            if size(img, 3) == 3
                img = rgb2gray(img);
            end
            all_features = [all_features, get_features(img, max_level)];
        else
            error('Invalid mode specified.');
        end
    end
end

function features = get_features(img, max_level)
    features = [];
    step = 8; 
    bin_size = 12;  
    for level = 0:max_level
        num_cells = 2^level;
        cell_size = floor(size(img) / num_cells);
        for i = 0:num_cells-1
            for j = 0:num_cells-1
                x_range = (i*cell_size(1) + 1):((i+1)*cell_size(1));
                y_range = (j*cell_size(2) + 1):((j+1)*cell_size(2));
                sub_img = img(x_range, y_range);
                [~, SIFT_features] = vl_dsift(single(sub_img), 'fast', 'step', step, 'size', bin_size);
                features = [features, SIFT_features];
            end
        end
    end
end

function image_feats = extract_spatial_pyramid_feats(image_paths, vocab, vocab_size, max_level)

    num_images = length(image_paths);
    num_bins = (4^(max_level+1)-1)/3;
    image_feats = zeros(num_images, vocab_size * num_bins);

    for i = 1:num_images
        img = single(imread(image_paths{i}));
        %img = rgb2gray(img);
        features = get_features(img, max_level);
        features = double(features);  


        if size(features, 1) ~= size(vocab, 1)
            features = features';
        end

        D = vl_alldist2(features, double(vocab));

        [~, min_idx] = min(D, [], 2);
        feature_vector = zeros(1, vocab_size * num_bins);
        index_offset = 0;

        for level = 0:max_level
            num_cells = 2^level;
            for cell = 1:num_cells
                idx_range = index_offset + (1:vocab_size);
                feature_vector(idx_range) = histcounts(min_idx((cell-1)*length(min_idx)/num_cells+1:cell*length(min_idx)/num_cells), 'BinEdges', 0.5:(vocab_size+0.5));
                %feature_vector(idx_range) = histc(min_idx((cell-1)*size(features,2)/num_cells+1:cell*size(features,2)/num_cells), 1:vocab_size);
                %feature_vector(idx_range) = histcounts(min_idx((cell-1)*numel(min_idx)/num_cells+1:cell*numel(min_idx)/num_cells), 1:vocab_size);
                index_offset = index_offset + vocab_size;
            end
        end

        feature_vector = feature_vector / sum(feature_vector);
        image_feats(i, :) = feature_vector;
    end
end

