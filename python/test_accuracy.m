addpath('D:\users\v-guoxie\data\ILSVRC2012\ILSVRC2012_devkit_t12\evaluation')
meta_file = 'D:\users\v-guoxie\data\ILSVRC2012\ILSVRC2012_devkit_t12\data/meta.mat';

ground_truth_file='D:\users\v-guoxie\data\ILSVRC2012\ILSVRC2012_devkit_t12\data/ILSVRC2012_validation_ground_truth.txt';
num_predictions_per_image=5;
root_path = '\\GCR/scratch/B99/v-guoxie/proto/residual_net/snapshot/';
% root_path  ='D:/users/v-guoxie/proto/vgg_A/snapshot/';
% root_path = 'D:\users\v-guoxie\work_place/result/test_03/';
% root_path = 'd:/users/v-guoxie/work_place/result/test_01/';
% root_path = '\\GCR/scratch/B99/v-guoxie/proto/multi_scale_alex/snapshot/log/'
files = dir([root_path '*.txt']);
% 
pred_files ={}
for i = 1:length(files)
    pred_files{i} = [root_path sprintf('residual_net_iter_%d.txt',i*10000)];%s
%     pred_files{i} = [root_path,files(i).name];
%     pred_files{i} = [root_path sprintf('alexnet_train_ms_iter_%d.txt',i*10000)];
end
% pred_files{1} = [root_path 'multi_scale_384_dense_ms.txt'];
% pred_files{1} = [root_path 'residual_net_iter_220000.txt'];
% pred_files = {[root_path 'ms_co_256.txt'],[root_path 'alex256x256_co.txt'],[root_path 'alex256_co.txt'],...
%     [root_path 'alex256x256_de.txt'],[root_path 'alex384_co.txt']};
load(meta_file);

%%% Task 1
result_name = [root_path 'result'];
fid = fopen(result_name,'a+');
all_top_1 = [];
for j=1:length(pred_files)
    pred_file=pred_files{j};
    error_flat=zeros(num_predictions_per_image,1);
    
    for i=1:num_predictions_per_image
        error_flat(i) = eval_flat(pred_file,ground_truth_file, i);
    end
    all_top_1 = [all_top_1 error_flat(1)];
    %disp('Task 1: # guesses  vs flat error');
    fprintf(fid,'Task 1: # guesses  vs flat error\n');
    fprintf(fid,[pred_file '\n']);
    fprintf(fid,'%d\t%f\n',[(1:num_predictions_per_image);error_flat']);
end
all_top_1
fclose(fid);
plot((1:length(pred_files))*10000,1-all_top_1);
title('test accuracy');
xlabel('iteration');
ylabel('error');
legend({'residual-net'},'Location','northwest');
saveas(gcf,[root_path 'test_accuracy.png']);
save([root_path 'info.mat'],'all_top_1');
%%
meta_file = 'E:\v-guoxie\data\ILSVRC2012\ILSVRC2012_devkit_t12\data/meta.mat';
load(meta_file);
names = {};
for i=1:1000
    names{i} = synsets(i).WNID;
end
[s_names,s_idx] = sort(names);
fid = fopen('synset_sort.txt','w');
for i=1:1000
fprintf(fid,'%d\n',s_idx(i));
end
fclose(fid);

%%
addpath('E:\v-guoxie\data\ILSVRC2012\ILSVRC2012_devkit_t12\evaluation')
meta_file = 'E:\v-guoxie\data\ILSVRC2012\ILSVRC2012_devkit_t12\data/meta.mat';
ground_truth_file='E:\v-guoxie\data\ILSVRC2012\ILSVRC2012_devkit_t12\data/ILSVRC2012_validation_ground_truth.txt';
pred_file1='result\test_01\alex_val_cls.txt';
pred_files={'result\test_07\select_combine.txt'}%{'result\test_01\replace_c5_122_c3_34.txt','result\test_01\replace_c5_228_c3_323.txt',...
%     'result\test_01\select_c5_122_c3_34.txt','result\test_01\select_c5_228_c3_323.txt'};%,...
%     'result\test_01\select_c5_122_c3_34_tune.txt'};

load(meta_file);
fid = fopen('val.txt','r');
img_list = textscan(fid,'%s %d');
img_list = img_list{1};
% img_root = 'E:\v-guoxie\data\ILSVRC2012\train\';
img_root = 'E:\v-guoxie\data\ilsvrc2012\val\';
% img_root2 = 'E:\v-guoxie\data\ilsvrc2012\val_256\';
fclose(fid);

for j=1:length(pred_files)
    pred_file2 = pred_files{j};
%%% Task 1
[error_flat1,c1] = eval_flat(pred_file1,ground_truth_file, 1);
[error_flat2,c2] = eval_flat(pred_file2,ground_truth_file, 1);
disp([error_flat1,error_flat2])
ca1 = logical(c1) & ~logical(c2);
ca2 = ~logical(c1) & logical(c2);

gt = dlmread(ground_truth_file);
org_fail = gt(ca1);
org_win = gt(ca2);

org_fail_hist = hist(org_fail,1:1000);
org_win_hist = hist(org_win,1:1000);
figure
bar(1:1000,org_fail_hist)
figure
bar(1:1000,org_win_hist)
[Y1,I1] =  sort(org_fail_hist,'descend');
[Y2,I2] = sort(org_win_hist,'descend');
org_fail_f = fopen([pred_file2(1:end-4) '_org_fail_cls.txt'],'w');
org_win_f = fopen([pred_file2(1:end-4) '_org_win_cls.txt'],'w');
for idx1=1:length(I1)
    fprintf(org_fail_f,'%d %d (%s)\n',I1(idx1),Y1(idx1),synsets(I1(idx1)).words);
end
for idx2=1:length(I2)
    fprintf(org_win_f,'%d %d (%s)\n',I2(idx2),Y2(idx2),synsets(I2(idx2)).words);
end
fclose(org_fail_f);
fclose(org_win_f);
% end

fail_samp = img_list(ca1);
mkdir([pred_file2(1:end-4) '_org_fail/']);
org_fail_list = fopen([pred_file2(1:end-1) '_org_fial_list.txt'],'w');
for i=1:length(fail_samp)
%     copyfile([img_root fail_samp{i}],[pred_file2(1:end-4) '_org_fail/' fail_samp{i}]);
    fprintf(org_fail_list,'%s\n',fail_samp{i});
end
fclose(org_fail_list);

fail_samp = img_list(ca2);
mkdir([pred_file2(1:end-4) '_org_win/']);
org_win_list = fopen([pred_file2(1:end-1) '_org_win_list.txt'],'w');
for i=1:length(fail_samp)
%     copyfile([img_root fail_samp{i}],[pred_file2(1:end-4) '_org_win/' fail_samp{i}]);
    fprintf(org_win_list,'%s\n',fail_samp{i});
end
fclose(org_win_list);
end

%%
% entropy
fid = fopen('vgg_bing_entropy.txt','r');
entropy_info = textscan(fid,'%d%f');
fclose(fid);
entropy_idx = entropy_info{1};
entropy_val = entropy_info{2};
hist(entropy_val)

sel_idx = entropy_val>6;
sel_entropy_idx = entropy_idx(sel_idx)+1;

fid = fopen('bing_list2.txt','r');
img_list = textscan(fid,'%s %d');
img_list = img_list{1};
% img_root = 'E:\v-guoxie\data\ILSVRC2012\train\';
img_root = '\\MSRA-SMS28\MSR-Bing Challenge on Image Retrieval Datasets\Train\image\';
fclose(fid);

save_path = 'un_represent_vgg_bing/';
if ~exist(save_path,'dir')
    mkdir(save_path)
end
for i=1:length(sel_entropy_idx)
    idx = sel_entropy_idx(i);
    name = img_list{idx};
    tmp_name = sprintf('%05d.jpg',i);
    copyfile([img_root '/' name],[save_path '/' tmp_name]);
end
%%
sel_entropy_idx = entropy_idx(end-length(sel_entropy_idx):end)+1;
sel_entropy_idx = sel_entropy_idx(end:-1:1);
save_path = 'represent_vgg_bing/';
if ~exist(save_path,'dir')
    mkdir(save_path)
end
for i=1:length(sel_entropy_idx)
    idx = sel_entropy_idx(i);
    name = img_list{idx};
    tmp_name = sprintf('%05d.jpg',i);
    copyfile([img_root '/' name],[save_path '/' tmp_name]);
end
%%
fid = fopen('vgg_pred.txt','r');
entropy_info = textscan(fid,'%d%f');
fclose(fid);
entropy_idx = entropy_info{1}+1;
entropy_val = entropy_info{2};
count = zeros(1,1000);
for i=1:length(entropy_idx)
count(entropy_idx(i)) = count(entropy_idx(i))+1;
end
% bar(count')
count = count / sum(count);
entropy = -count.*log(count);
figure;
bar(entropy)
sum(entropy)