function []=NormImg_script()
root_dir = '/path/to/dataset/train';
save_dir = '/path/to/dataset/train14';
dir1={'A03','A04','A05','A07','A10','A11','A12','A14','A15','A17','A18'};
for i = 1:length(dir1)
    imglist=dir(fullfile(root_dir, dir1{i},'*.tiff'));
    for j =1:length(imglist)	
        Norm(root_dir, dir1{i}, imglist(j).name, save_dir);
    end
end

function []= Norm(root_dir, dir1, imgname, save_dir)
img= fullfile(root_dir, dir1, imgname);
img_n= fullfile(save_dir, dir1, imgname);
im=imread(img);
[Inorm1 H1 E1] = normalizeStaining(im);
imwrite(Inorm1,img_n);
