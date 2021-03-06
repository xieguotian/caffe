function A = show_tiles(img_sets,varargin)
% img_sets: sets of images. matrix of [w,h,ch,sub_num,num]
ndim = ndims(img_sets);
if ndim>5
    error('dims must less than 5');
end
isscale = true;
if nargin==2
    isscale = varargin{1};
end
niter = ndim - 3;
pad = max(ceil(size(img_sets,1) / 20),1);
if niter==1
    A = permute(img_sets,[4,1,2,3]);
    A = show_sub_tiles(A,pad,isscale);
else
    A = [];
    for i=1:size(img_sets,5)
        tmpA = img_sets(:,:,:,:,i);
        tmpA = permute(tmpA,[4,1,2,3]);
        tmpA = show_sub_tiles(tmpA,pad,isscale);
        A = cat(4,A,tmpA);
    end
    A = permute(A,[4,1,2,3]);
    red = [255,0,0];
    A = show_sub_tiles(A,pad,isscale,red);
end

function nA = show_sub_tiles(A,pad, isscale,varargin)
% show color image
% conv1=permute(conv1,[4,1,2,3])
% show_tile(conv1)
% or
% show gray image
% conv1 = permute(conv1,[4,3,1,2])
% conv1 = reshape(conv1,size(conv1,1)*size(conv1,2),size(conv1,3),size(conv1,4));
% show_tile(conv1)
iscolor = false;
if nargin == 4
    iscolor = true;
    pad_value = varargin{1};
    %pad_value = reshape(pad_value,1,1,1,3);
end

if size(A,4)>1
    len = size(A,1);
    sq_len = ceil(sqrt(len));
    p_h = size(A,2);
    p_w = size(A,3);
    ch = size(A,4);
    % A as matrix n*h*w*ch
    A = reshape(A,len,p_h*p_w*ch);
    if isscale
        A = bsxfun(@minus,A,min(A,[],2));
        A = bsxfun(@rdivide,A,max(A,[],2));
    end
    A = reshape(A,len,p_h,p_w,ch);
    % re-arrange to add pad pixel
    nA = zeros(sq_len*sq_len,p_h+pad,p_w+pad,ch);
    if iscolor
        nA(:,:,:,1) = pad_value(1);
        nA(:,:,:,2) = pad_value(2);
        nA(:,:,:,3) = pad_value(3);
    end
    nA(1:len,1:p_h,1:p_w,1:ch) = A;
    out = [];
    % use col2im to re-arrange each ch
    for i=1:ch
        nA2 = reshape(nA(:,:,:,i),sq_len*sq_len,(p_h+pad)*(p_w+pad))';
        nA2 = col2im(nA2,[p_h+pad,p_w+pad],[(p_h+pad)*sq_len,(p_w+pad)*sq_len],'distinct');
        out=cat(3,out,nA2);
    end
    nA=out;
else
    len = size(A,1);
    sq_len = ceil(sqrt(len));
    p_h = size(A,2);
    p_w = size(A,3);
    % A as matrix n*h*w
    A = reshape(A,len,p_h*p_w);
    if isscale
        A = bsxfun(@minus,A,min(A,[],2));
        A = bsxfun(@rdivide,A,max(A,[],2));
    end
    A = reshape(A,len,p_h,p_w);
    % add pad pixel
    nA = zeros(sq_len*sq_len,p_h+pad,p_w+pad);
    nA(1:len,1:p_h,1:p_w) = A;
    % rearrange to (h*w)*n
    nA = reshape(nA,sq_len*sq_len,(p_h+pad)*(p_w+pad))';
    % use col2im to re-arrange the image
    nA = col2im(nA,[p_h+pad,p_w+pad],[(p_h+pad)*sq_len,(p_w+pad)*sq_len],'distinct');
end