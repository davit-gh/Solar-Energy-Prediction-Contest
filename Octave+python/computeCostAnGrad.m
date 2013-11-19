% Computes cost function and its gradient with respect to theta
function [J, grad] = computeCostAndGrad(theta,X, y)

% Initialize some useful values
m = length(y); % number of training examples
n = size(X,2);

grad = zeros(size(n));

for j=1:n,
	grad(j) = (1/m)*(X*theta-y)'*X(:,j);
end;

J = 1/2/m*(X*theta-y)'*(X*theta-y);

end

% Computes optimal 'theta' values using advance 
% optimization function fminunc
% Performs cross - valiation and outputs average MAE
function [avgmae theta] = cvscore(trainFileName)
X=csvread(trainFileName);
x=X(:,[2:16]);
y=X(:,17);
mn=mean(x);
sigma=std(x);
x=bsxfun(@minus,x,mn);
x=bsxfun(@rdivide,x,sigma);
x=[ones(size(x,1),1) x];
init_theta=zeros(size(x,2),1);
[J, grad] = computeCostAndGrad(init_theta, x, y);
options = optimset('GradObj', 'on', 'MaxIter', 400);
holdoutsize=size(x,1)*0.3;
n=10;
avgmae = 0;
for i=1:n
	cvrp = randperm(holdoutsize);
	cvTrain = setdiff([1:size(x,1)],cvrp);
	cvX=x(cvrp,:);
	cvY=y(cvrp);
	trainX = x(cvTrain,:);
	trainY = y(cvTrain);
	[theta cost] = fminunc(@(t)(computeCostAndGrad(t,trainX,trainY)),init_theta,options);
	hx = cvX * theta;
	mae = sum(abs(cvY - hx))/size(cvY,1);
	avgmae += mae;
end;
avgmae = avgmae/n
end
