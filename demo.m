clear all
clc
close all

load exampledata
%% load training data
Input1.LabeledData=Data_Train; %% labelled training data
Input1.GroundTruth=Label_Train; %% labels
Input1.UnlabeledData=Data_Test; %% unlabeled training data
Input1.LayerNum=3; %% 
Input.Lambda=1.1;
%% run semi-supervised learning
[Output1]=STHP(Input1,'ssl');
%% load testing data
Input2.UnlabeledData=Data_Test; %% unlabeled training data
Input2.Syst=Output1.Syst; 
%% run testing
[Output2]=STHP(Input2,'ta');
%% calculate the classification accuracy
ConfusionMatrix=confusionmat(Label_Test,Output2.Labels);
Ltemp=size(ConfusionMatrix,1);
Acc2=sum(sum(ConfusionMatrix.*eye(Ltemp)))/sum(sum(ConfusionMatrix))
