function process_confidence_dir(imdir, outdir, ext, name)

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
        if size(im, 3) == 1
            im = repmat(im, [1 1 3]);
        end
        [det, all, info, model_score] = process(im, model, threshold);
        parsave(conf_file, det, model_score);
    end
end

matlabpool close;

function parsave(filename, det, model_score)

save(filename, 'det', 'model_score');