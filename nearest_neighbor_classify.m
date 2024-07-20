% %%
% 
function predicted_categories = nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, k, distance_measure)
    if nargin < 4
        k = 13;
    end
    N = size(test_image_feats, 1);
    M = size(train_image_feats, 1);
    predicted_categories = cell(N, 1);

    if k > M
      k = M;
    end

    if nargin < 5 || strcmp(distance_measure, 'euclidean')
        distances = zeros(N, M);
        for i = 1:N
            for j = 1:M
                diff = test_image_feats(i, :) - train_image_feats(j, :);
                distances(i, j) = sqrt(sum(diff .^ 2));
            end
        end
    elseif strcmp(distance_measure, 'cosine')
        distances = zeros(N, M);
        for i = 1:N
            for j = 1:M
                dot_product = dot(test_image_feats(i, :), train_image_feats(j, :));
                norm_product = norm(test_image_feats(i, :)) * norm(train_image_feats(j, :));
                distances(i, j) = 1 - dot_product / norm_product;
            end
        end
    elseif strcmp(distance_measure, 'manhattan')
        distances = zeros(N, M);
        for i = 1:N
            for j = 1:M
                diff = abs(test_image_feats(i, :) - train_image_feats(j, :));
                distances(i, j) = sum(diff);
            end
        end
    elseif strcmp(distance_measure, 'chebyshev')
        distances = zeros(N, M);
        for i = 1:N
            for j = 1:M
                diff = abs(test_image_feats(i, :) - train_image_feats(j, :));
                distances(i, j) = max(diff);
            end
        end
    else
        error('Unsupported distance measure');
    end

    for i = 1:N
        [~, sortedIndices] = sort(distances(i, :));

        nearestLabels = train_labels(sortedIndices(1:k));
        [uniqueLabels, ~, idx] = unique(nearestLabels);
        labelCounts = accumarray(idx, 1);
        [~, maxIdx] = max(labelCounts);
        predicted_categories{i} = uniqueLabels{maxIdx};
    end
end
