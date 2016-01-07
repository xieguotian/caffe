function top_imgs = filter_img_vis(net_param,layer_idx,samp_idx,img_list,img_root)
% net_param: a matrix contain net layer info
% layer_idx: index of layer in the net_param
% samp_idx: index of neuron that to visualize
% img_list: name of images
% img_root: root path of images set.

% get net info
net_info = caffe.receptive_field_info(net_param);
kernel = net_info(layer_idx,1);
stride = net_info(layer_idx,2);
pad = net_info(layer_idx,3);

% image visualizations size
top_num = size(samp_idx,2)/3.0;
% if kernel > 100
    img_size = 100;
% else
%     img_size = kernel;
% end

% get images
top_imgs = zeros(img_size,img_size,3,top_num,size(samp_idx,1));
for i=1:size(samp_idx,1)
    for j=1:3:size(samp_idx,2)
        % read image
        img_idx = samp_idx(i,j);
        im_name = [img_root '/' img_list{img_idx}];
        img = imread(im_name);
        im_size = [size(img,1),size(img,2)];
        % receptive field
        w_idx = samp_idx(i,j+2);
        h_idx = samp_idx(i,j+1);

        h_st = max((h_idx-1)*stride+pad,0)+1;
        w_st = max((w_idx-1)*stride+pad,0)+1;
        h_end = min(h_st+kernel-1,im_size(1));
        w_end = min(w_st+kernel-1,im_size(2));
        
        % save image
        img = img(h_st:h_end,w_st:w_end,:);
        if img_size~=kernel
            img = imresize(img,[img_size, img_size]);
        end
        idx = floor(j/3)+1;
        if size(img,3)==1
            top_imgs(1:size(img,1),1:size(img,2),:,idx,i) = cat(3,img,img,img);
        else
            top_imgs(1:size(img,1),1:size(img,2),:,idx,i) = img;
        end
    end
end