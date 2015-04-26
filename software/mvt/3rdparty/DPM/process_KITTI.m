% process the KITTI dataset
function process_KITTI

src_path = '/home/yuxiang/Projects/Multiview_Tracking/multiview_tracking_dataset/KITTI';
dst_path = '/home/yuxiang/Projects/Multiview_Tracking/result/DPM_VOC2007';
dir_names = {'01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11'};
ext = 'png';

N = numel(dir_names);
for i = 1:N
    disp(dir_names{i});
    src_dir = fullfile(src_path, dir_names{i}, 'img');
    dst_dir = fullfile(dst_path, dir_names{i});
    if exist(dst_dir, 'dir') == 0
        mkdir(dst_dir);
    end
    process_confidence_dir(src_dir, dst_dir, ext, 'car_voc2007.mat');
end