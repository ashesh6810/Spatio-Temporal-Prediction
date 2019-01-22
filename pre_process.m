clear all;
close all;
clc;

load('Lorenz96_train_F_4.mat');
X=(X-mean(X(:)))/std(X(:));
leadday=1;
training_samples=30000;
train_input=X(:,1:training_samples);
train_label=X(:,leadday+1:training_samples+leadday);

test_input=X(:,training_samples+leadday+1:end-leadday);
test_labels=X(:,training_samples+2*leadday+1:end);


csvwrite(['train_input' num2str(leadday) '.csv'],train_input);
csvwrite(['train_label' num2str(leadday) '.csv'],train_label);

csvwrite(['test_input' num2str(leadday) '.csv'],test_input);
csvwrite(['test_labels' num2str(leadday) '.csv'],test_labels);



