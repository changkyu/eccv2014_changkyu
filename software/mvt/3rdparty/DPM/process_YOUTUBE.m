% process the YOUTUBE dataset
function process_YOUTUBE

src_path = '/home/yuxiang/Projects/Multiview_Tracking/multiview_tracking_dataset/YOUTUBE';
dst_path = '/home/yuxiang/Projects/Multiview_Tracking/result/DPM_VOC2007';
dir_names = {'race1', 'race2', 'race3', 'race4', 'race5', 'race6', 'sedan', 'SUV1', 'SUV2'};
exts = {'png', 'png', 'png', 'png', 'png', 'png', 'jpg', 'jpg', 'jpg'};

N = numel(dir_names);
for i = 1:N
    disp(dir_names{i});
    src_dir = fullfile(src_path, dir_names{i}, 'img');
    dst_dir = fullfile(dst_path, dir_names{i});
    process_confidence_dir(src_dir, dst_dir, exts{i}, 'car_voc2007.mat');
end