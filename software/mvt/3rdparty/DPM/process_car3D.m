function process_car3D

imdir = '/home/yuxiang/Projects/Multiview_Tracking/dataset/car_3D/img';
outdir = '/home/yuxiang/Projects/Multiview_Tracking/result/DPM_VOC2007/car_3D';
ext = 'jpg';
name = 'car_voc2007';

% load model
data = load(name);
model = data.model;
threshold = -2;

if ~exist(outdir, 'dir')
	mkdir(outdir);
end

matlabpool open;
files = dir([imdir '/*.' ext]);
for idx = 1:16:length(files)
    parfor j = 1:16
        i = idx + j - 1;
        if(i > length(files))
            continue;
        end        
        filename = [imdir '/' files(i).name];
        conf_file = [outdir '/' files(i).name(1:end-4) '.mat'];

        disp(['process ' filename]);

        if(~exist(filename)) 
            disp(['file ' filename ' doesnot exist?\n'])
            continue;
        end

        if(exist(conf_file))
            continue;
        end

        % run LSVM detector
        im = imread(filename);
        [det, all, info, model_score] = process(im, model, threshold);
        parsave(conf_file, det, model_score);
    end
end

matlabpool close;

function parsave(filename, det, model_score)

save(filename, 'det', 'model_score');