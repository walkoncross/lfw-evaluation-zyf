%img_root_dir = 'C:/datasets/lfw-aligned/'

list_file = '../lfw_data/lfw_list_mtcnn.txt';
pairs_file = '../lfw_data/lfw_pairs.txt'

% read list_file into list_imgs and list_ids
fid = fopen(list_file);
lines = textscan(fid,'%s %d');
list_imgs = lines{1};
list_ids = lines{2};
fclose(fid);

% open pairs_file
fid = fopen(pairs_file);
CC = fscanf(fid,'%d %d');
n_set = CC(1);n_num=CC(2);

same_pair = cell(n_set*n_num,2);
diff_pair = cell(n_set*n_num,2);
lfw_label = zeros(n_set*n_num * 2,2);

for i=1:n_set
    for j = 1 : n_num
        CC = textscan(fid, '%s %d %d\n', 1);
        p = CC{1};id1=CC{2};id2=CC{3};
        file1 = sprintf('%s/%s_%04d.jpg', p{1},p{1},id1);
        file2 = sprintf('%s/%s_%04d.jpg', p{1},p{1},id2);
        same_pair((i-1)*n_num + j,1) = {file1};
        same_pair((i-1)*n_num + j,2) = {file2};
        if exist('list_imgs','var')
            lfw_label((i-1)*n_num + j,1) = find(strcmp(list_imgs, file1));
            lfw_label((i-1)*n_num + j,2) = find(strcmp(list_imgs, file2));
        end;
    end;
    for j = 1 : n_num
        CC = textscan(fid, '%s %d %s %d\n', 1);
        p1 = CC{1};id1=CC{2};p2=CC{3};id2=CC{4};
        file1 = sprintf('%s/%s_%04d.jpg',p1{1},p1{1},id1);
        file2 = sprintf('%s/%s_%04d.jpg',p2{1},p2{1},id2);

        diff_pair((i-1)*n_num + j,1) = {file1};
        diff_pair((i-1)*n_num + j,2) = {file2};
        if exist('list_imgs','var')
            lfw_label(n_set*n_num + (i-1)*n_num + j,1) = find(strcmp(list_imgs, file1));
            lfw_label(n_set*n_num + (i-1)*n_num + j,2) = find(strcmp(list_imgs, file2));
        end;
    end;
end;
fclose(fid);

pos_pair_zyf = lfw_label(1:3000, 1:2)';
neg_pair_zyf = lfw_label(3001:end, 1:2)';

if exist('feature','var')
    AllFeature1 = feature(:,lfw_label(:,1));
    AllFeature2 = feature(:,lfw_label(:,2));
end;