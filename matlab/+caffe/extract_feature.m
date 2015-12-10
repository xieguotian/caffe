function feat = extract_feature(net_proto,net_param,blob_name,num_batch,device_id)
% params:
%   net_proto(string): network prototxt file name
%   net_param(string): network parameter file name
%   blob_name(string): blobs names that are to extract feature
%   num_batch(integer): number of batches
%   device_id(string): use cpu or gpu

% set mode of gpu or cpu
if device_id=='gpu'
    caffe.set_mode_gpu()
else
    caffe.set_mode_cpu()
end

% load network and parameter
disp('load net');
net = caffe.Net(net_proto,net_param,'test');

% initial feat vector
disp('extract feature...')
feat=cell(1,length(blob_name));
for i=1:length(feat)
    feat{i} = [];
end

% extract feature for each batch
input_data={};
for i=1:num_batch
    net.forward(input_data);
    fprintf('batch %d\n',i);
    for j=1:length(blob_name)
        feat_data = net.blobs(blob_name{j}).get_data();
        feat{j} = cat(4,feat{j},feat_data);
    end
end


