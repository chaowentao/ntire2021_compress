import os

path = '/root/ntire/Data/video_compress_track2/test_youtube/images/train/'

# 获取该目录下所有文件，存入列表中
filenames = os.listdir(path)
filenames.sort(key=lambda x: int(x), reverse=False)
print(filenames[:10])
count = 1
for filename in filenames:
    # 设置旧文件名（就是路径+文件名）
    oldname = os.path.join(path, filename)  # os.sep添加系统分隔符

    # 设置新文件名
    newname = os.path.join(path, f'{int(count):03d}')

    os.rename(oldname, newname)  #用os模块中的rename方法对文件改名
    print(oldname, '======>', newname)
    count += 1

root_path = '/root/ntire/Data/video_compress_track2/test_youtube/images/train/'
for i in range(1, 18):
    path = os.path.join(root_path, f'{int(i):03d}')
    # 获取该目录下所有文件，存入列表中
    filenames = os.listdir(path)
    filenames.sort(key=lambda x: int(x[:-4]), reverse=False)
    print(filenames[:10])
    count = 1
    for filename in filenames:
        # 设置旧文件名（就是路径+文件名）
        oldname = os.path.join(path, filename)  # os.sep添加系统分隔符

        # 设置新文件名
        newname = os.path.join(path, f'{int(count):03d}.png')

        os.rename(oldname, newname)  #用os模块中的rename方法对文件改名
        print(oldname, '======>', newname)
        count += 1
