%% Instructions
%  Using pca to reduce dimension and use logistic regression to divide two classes.  
%  At last , using examples to test the model
%% Initialization
clear ; close all; clc

%% Part 1: Load Training Set
% Load the training set
% There are 4 labels:stature,weight,length of hair,sex

load('data1.mat');
X = data1;
X = X(:,1:3);


% Visualize the example
scatter3(X(:,1),X(:,2),X(:,3),'x');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Part 2: Principal Component Analysis

[X_norm, mu, sigma] = featureNormalize(X);

%Visualize the example
scatter3(X_norm(:,1),X_norm(:,2),X_norm(:,3),'x');

%  Run PCA
%  In this step , use svd function
[U, S, V] = pca(X_norm);

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Part 3: Dimension Reduction

%The dimension we want,K is the dimension after dimension reduction
K = 2;

U_reduce = U(:, 1:K);
%Z is the data after dimension reduction
Z = X_norm * U_reduce;

%Visualize the example
plot(Z(:,1),Z(:,2),'x');

fprintf('Program paused. Press enter to continue.\n');
pause;


%% Part 4: Logistic Regression

X = Z(:, [1, 2]); y = data1(:, 4);
plotData(X,y);


X = mapFeature(X(:,1), X(:,2));

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1
lambda = 1;

% Compute and display initial cost and gradient for regularized logistic
% regression
[cost, grad] = costFunctionReg(initial_theta, X, y, lambda);

fprintf('Cost at initial theta (zeros): %f\n', cost);

% Initialize fitting parameters
initial_theta = zeros(size(X, 2), 1);

% Set regularization parameter lambda to 1 (you should vary this)
lambda = 1;

% Set Options
options = optimset('GradObj', 'on', 'MaxIter', 400);

% Optimize
[theta, J, exit_flag] = ...
	fminunc(@(t)(costFunctionReg(t, X, y, lambda)), initial_theta, options);

% Plot Boundary
plotDecisionBoundary(theta, X, y);
hold on;
title(sprintf('lambda = %g', lambda))

% Labels and Legend

legend('y = 1', 'y = 0', 'Decision boundary')

fprintf('Program paused. Press enter to continue.\n');
pause;

%% Part 5: Test Example

load 'data2.mat'
X_ex = data2(:,1:3);y_ex = data2(:,4);
[X_norm, mu, sigma] = featureNormalize(X_ex);
[U, S] = pca(X_norm);

U_reduce = U(:, 1:K);
Z = X_norm * U_reduce;
plotData_ex(Z,y_ex);

