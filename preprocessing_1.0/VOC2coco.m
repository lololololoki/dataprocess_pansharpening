% USAGE
%  CocoUtils.convertPascalGt( dataDir, year, split, annFile )
%
% INPUTS
%  dataDir    - dir containing VOCdevkit/
%  year       - dataset year (e.g. '2007')
%  split      - dataset split (e.g. 'val')
%  annFile    - annotation file for writing results

clear;
clc;

addpath(genpath('../MatlabAPI'));
addpath(genpath('../VOCcode'));
addpath(genpath('../VOCdevkit'));

dataDir = ['../VOCdevkit'];
year = ['0712'];
splits = {'test','trainval'};
for i=1:2
    split = splits{i};
    annFile = [dataDir,'/VOC0712/voc_' year '_' split '.json'];
    CocoUtils_building.convertPascalGt( dataDir, year, split, annFile );
end

