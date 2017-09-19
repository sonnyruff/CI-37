clear all
close all

theta = -1;
amount = 1000;

% Training input part
%in = dlmread(fullfile('C:','Users','Thomas van der Pas','Documents','TU Delft', 'Vakken', 'Computational Intelligence','CI-37','data','features.txt'));
in = dlmread("data/features.txt");
input = in(1:amount,:);

% unrandomize input:
% input = in(1,:);
% for i=1:15
%     input(i,:) = input(1,:);
% end

n1 = size(input);


%wantedOut = dlmread(fullfile('C:','Users','Thomas van der Pas','Documents','TU Delft', 'Vakken', 'Computational Intelligence','CI-37','data','targets.txt'));
wantedOut = dlmread("data/targets.txt");
wantedOutput = wantedOut(1:amount,:);


n2 = max(wantedOutput);
out = full(ind2vec(wantedOutput',n2))';

output = zeros(1,n2);


siz = size(wantedOutput);

vectors = full(ind2vec(wantedOutput',7))';

errorSumSquared = [];

% Learning rate
a = 0.1;

% Initial weights
w = 2.*rand(n1(2),n2) - 1;

% unrandomize w
% w = 2.*rand(n1(2),1) - 1;
% for i=1:n2
%     w(:,i) = w(:,1);
% end

solutionFound = false;
iter = 1;
epoch = 1;
error = zeros(n1(1),n2);


disp('w_begin');
disp(w);
for iter = 1:n1(1)
    for i = 1:n2
        output(iter,i) = perceptron(input(iter,:), w(:,i),theta); 
%         disp("Output = ");
%         disp(output);
        
        error(iter,i) = out(iter,i)-output(iter,i);
%         disp('Error = ');
%         disp(error);
        
        deltaw(:,1) = a*input(iter,:)*error(iter,i);
%         disp('DeltaW = '); 
%         disp(deltaw(:, 1));
        
        w(:,i) = w(:,i) + deltaw(:,1);
%         disp('w = ');
%         disp(w);
%         
%         pause;
    end
    
end

disp('w_eind');
disp(w);



% while(~solutionFound)
%     for i = 1:n2
%         output(i) = perceptron(input(iter,:), w(iter,:));
% 
%         error(iter,i) = abs(out(iter,i) - output(i));
%     end
%     if ~(output == out(iter))
%         errorSumSquared = sumsqr(error);
%     end
%     solutionFound = true;
% end
%     
% 
% 
% % while(~solutionFound)
% %     for i = 1:n2
% %         output(i) = perceptron(input(i,:), w(iter,:));
% %         error(iter) = out(iter,i) - output(i);
% %         w(iter + 1,:) = w(iter,:) + a * input(i,:) * error(iter);
% %         iter = iter + 1;
% %     end
% %     errorSumSquared(epoch) = sumsqr(error);
% %     
% %     disp(epoch + " : ") 
% %     for i=1:n2
% %         fprintf(output(i) + ", ");
% %     end
% %         
% %     
% % end
% 
% 
% % %     for i = 1:n(2)
% % %         output(i) = perceptron(input(i,:), w(iter,:));
% % %         
% % %         error(iter) = wantedOutput(i) - output(i);
% % %         w(iter + 1,:) = w(iter,:) + a * input(i,:) * error(iter);
% % %         iter = iter + 1;
% % %     end
% % %     errorSumSquared(epoch) = sumsqr(error);
% % %     
% % %     disp(epoch + " : " + output(1) + ", " + output(2) + ", " + output(3) + ", " + output(4));
% % %     
% % %     epoch = epoch + 1;
% % %     
% % %     if output == wantedOutput | iter > 10000
% % %         solutionFound = true;
% % %     end
% % % end
% % % 
% % % plot(errorSumSquared);
% % % figure
% % % plot(w);
