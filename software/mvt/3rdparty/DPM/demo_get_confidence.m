function demo_get_confidence

% load the detections and model from DPM
im = imread('000034.jpg');
object = load('000034.mat');
det = object.det;
model_score = object.model_score;

% decide the location to get confidences
% let use the left corner of det(1,:)
x = det(1,1);
y = det(1,2);

% call the get_confidence function
[dets, boxes, info] = get_confidence(model_score, x, y);

% the dets are now sorted from smallest scale to largest scale
% sort the dets by confidence
[~, index] = sort(dets(:,6), 'descend');
dets = dets(index,:);

% show the highest score bounding boxes
showboxes(im, dets(1,:));

