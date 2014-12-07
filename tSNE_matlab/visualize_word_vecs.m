
% L is a  |V| x n matrix
%L = dlmread('../data/wordMatrices/origWordVectors.txt');
learnedL = dlmread('../data/wordMatrices/learnedWordVectors.txt');


indecesInLandLabels = dlmread('trainData_indeces_and_labels.txt');
indeces = indecesInLandLabels(:,1)+1;
labels = indecesInLandLabels(:,2);

%relevantL = L(indeces, :);
relevantLearnedL = learnedL(indeces, :);

no_dims = 2;
init_dims = 30;
perplexity = 30;

% function ydata = tsne(X, labels, no_dims, initial_dims, perplexity)
%mappedX = tsne(relevantL, [], no_dims, init_dims, perplexity);
mappedX = tsne(relevantLearnedL, [], no_dims, init_dims, perplexity);
dlmwrite('./tsne_vecs_learned_L.txt', mappedX);
