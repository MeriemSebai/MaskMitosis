%% segmentation of the remaining mitosis ground truths with active contours method

root_dir = 'train_patches';
save_dir = 'train_patches_segmented';
if ~exist(save_dir, 'dir')
   mkdir(save_dir);
end
imglist=dir(fullfile(root_dir,'A*'));
for j =1:length(imglist)
 	disp(imglist(j).name);
	I = imread(fullfile(root_dir,imglist(j).name));  %-- load the image
	r=double(I(:,:,1));
	g=double(I(:,:,2));
	b=double(I(:,:,3));
	blue_ratio=(((100*b)./(1+r+g)).*(256./(1+b+r+g)));  %-- transforming the RGB image into BR image
	
	m = zeros(size(I,1),size(I,2));          %-- create initial mask
	C = strsplit(imglist(j).name,'_');
	d1=str2num(C{3});
	d2=str2num(C{4});
	d3=str2num(C{5});
	d4=str2num(C{6});
	x=30
	if d1==0 
		w1=x
	else if d1>x
			w1=1
             else w1=abs(d1-20)
	     end
	end

	if d2==0
		w2=size(I,2)-x
	else if d2>x 
		w2=size(I,2)
	     else w2=size(I,2)-(x-d2)
	     end
        end	
                	
	if d3==0
		w3=x
	else if d3>x
		w3=1
	     else w3=abs(d3-20)
	     end	
	end

	if d4==0
		w4=size(I,1)-x
	else if d4>x
                w4=size(I,1) 
             else w4=size(I,1)-(x-d4)
	     end	
	end
	disp(size(I,2))
        disp(size(I,1))	
	m(w3:w4,w1:w2) = 1;
	
	seg = region_seg(blue_ratio, m, 600) %-- Run segmentation
	seg=bwareaopen(seg,50);
	disp('done');
	
	bw=im2bw(seg,0.275); 
	imwrite(bw,fullfile(save_dir,imglist(j).name));

end


