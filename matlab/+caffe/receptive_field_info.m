function info = receptive_field_info(param)
% param is a matrix, with each row as a layer param
% each element of row is kernel size, stride, pad.
% for eample, alex net:
%     k,s,p
% param = ...
%    [11,4,0;... % conv1
%     3,2,0;...       % pool1
%     5,1,2;...       % conv2
%     3,2,0;...       % pool2
%     3,1,1;...       % conv3
%     3,1,1;...       % conv4
%     3,1,1;...       % conv5
%     3,2,0];         % pool5

info = zeros(size(param,1),3);

info(1,1) = param(1,1);
info(1,2) = param(1,2);
info(1,3) = -param(1,3);
for i=2:size(info,1)
    info(i,1) = (param(i,1)-1)*info(i-1,2)+info(i-1,1);
    info(i,2) = param(i,2)*info(i-1,2);
    info(i,3) = info(i-1,3) - param(i,3)*info(i-1,2);
end
