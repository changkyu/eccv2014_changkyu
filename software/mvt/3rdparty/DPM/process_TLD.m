% process the TLD dataset
function process_TLD

src_path = '/home/yuxiang/Projects/Multiview_Tracking/multiview_tracking_dataset/TLD';
dst_path = '/home/yuxiang/Projects/Multiview_Tracking/result/DPM_VOC2007';
dir_names = {'06_car'};
exts = {'jpg'};

N = numel(dir_names);
for i = 1:N
    disp(dir_names{i});
    src_dir = fullfile(src_path, dir_names{i}, 'img');
    dst_dir = fullfile(dst_path, dir_names{i});
    process_confidence_dir(src_dir, dst_dir, exts{i}, 'car_voc2007.mat');
end