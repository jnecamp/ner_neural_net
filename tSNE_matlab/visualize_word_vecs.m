
% L is a  |V| x n matrix
L = dlmread('../data/wordMatrices/origWordVectors.txt');


relevantIndices = dlmread('relevant_indices.txt');
relevantIndices = round(relevantIndices + 1);

newL = L(relevantIndices, :);

no_dims = 2;
init_dims = 30;
ratio_landmarks = .1;
perplexity = 30;

% function ydata = tsne(X, labels, no_dims, initial_dims, perplexity)
mappedX = tsne(newL, [], no_dims, init_dims, perplexity);
