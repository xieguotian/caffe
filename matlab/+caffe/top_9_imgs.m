function [imgs, top_imgs_dis] = top_9_imgs(feat,net_param,layer_idx,img_list,img_root)

feat_dis = zeros(size(feat{1},3),length(feat));
feat_idx = zeros(size(feat{1},3),length(feat));

for i=1:length(feat)
    f = feat{i};
    [w,h,ch] = size(f);
    f = reshape(f,w*h,ch);
    [feat_dis(:,i),feat_idx(:,i)] = max(f,[],1);
end

[f_v,f_i] = sort(feat_dis,2,'descend');
feat_sample_idx = zeros(size(f_i,1),9*3);

for i=1:size(feat_sample_idx,1)
    for j=1:9
        [w,h,ch] = size(feat{f_i(i,j)});
        idx = feat_idx(i,f_i(i,j));
        [w_idx,h_idx] = ind2sub([w,h],idx);
        
        feat_sample_idx(i,(j-1)*3+1) = f_i(i,j);
        feat_sample_idx(i,(j-1)*3+2) = h_idx;
        feat_sample_idx(i,(j-1)*3+3) = w_idx;
    end
end
imgs = caffe.filter_img_vis(net_param,layer_idx,feat_sample_idx,img_list,img_root);
top_imgs_dis.feat_dis = feat_dis;
top_imgs_dis.feat_idx = feat_idx;