import os
import cv2
# root_path = '/root/ntire/Data/video_compress_track2/train_expand_199_2/gt/'

# root_path = '/root/ntire/Data/video_compress_track2/test_youtube/images/train_raw/'
root_path = '/root/ntire/Data/video_compress_track3/images/test/'
foldernames = os.listdir(root_path)
foldernames.sort()
print(foldernames[:5])
txt_name = 'meta_info_Compress_test.txt'
txt_file = os.path.join('/root/ntire/Data/video_compress_track3/', txt_name)
with open(txt_file, 'w') as f:
    for foldername in foldernames:
        print(foldername)
        floder_path = os.path.join(root_path, foldername)
        filenames = os.listdir(floder_path)
        filenames.sort()
        file_path = os.path.join(floder_path, filenames[0])
        image = cv2.imread(file_path)
        height, width, channel = image.shape[:]
        print(height, width, channel)
        for filename in filenames:
            frame = filename.split('.')[0]
            f.write(
                f'{int(foldername):03d}/{int(frame):03d}.png ({height}, {width}, {channel})\n'
            )

# print(f'Generate annotation files {file_name}...')
# txt_file = osp.join(root_path, file_name)
# mmcv.utils.mkdir_or_exist(osp.dirname(txt_file))
# xls_file = osp.join(root_path, xls_name)
# workbook = xlrd.open_workbook(xls_file)  # 文件路径
# worksheet = workbook.sheet_by_index(0)

# nrows = worksheet.nrows  #获取该表总行数
# print(nrows)  # 200

# ncols = worksheet.ncols  #获取该表总列数
# print(ncols)  # 5
# with open(txt_file, 'w') as f:
#     for i in range(1, nrows):
#         frames = int(worksheet.cell_value(i, 3))
#         print(i, frames)
#         for j in range(1, frames + 1):
#             f.write(f'{i:03d}/{j:03d}.png (536, 960, 3)\n')
