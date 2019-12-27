#encoding:utf-8
# 将tif格式转位jpg 
import os 
from PIL import Image 
import shutil 
import sys 
  
# # Define the input and output image 
# output_dirHR = '../data/Mosaic_HR/'
# output_dirLR = '../data/Mosaic_LR/'
# if not os.path.exists(output_dirHR): 
  # os.mkdir(output_dirHR) 
# if not os.path.exists(output_dirLR): 
  # os.mkdir(output_dirLR) 
  
output_dir = '../jpg/'
if not os.path.exists(output_dir): 
  os.mkdir(output_dir) 
  
  
  
def image2jpg(dataset_dir,type): 
  files = [] 
  image_list = os.listdir(dataset_dir) 
  files = [os.path.join(dataset_dir, _) for _ in image_list] 
  for index,jpg in enumerate(files): 
    # if index > 100000: 
      # break
    try: 
      sys.stdout.write('\r>>Converting image %d/100000 ' % (index)) 
      sys.stdout.flush() 
      im = Image.open(jpg) 
      jpg = os.path.splitext(jpg)[0] + "." + type
      im.save(jpg) 
      # 将已经转换的图片移动到指定位置 
      ''''' 
      if jpg.split('.')[-1] == 'jpg': 
        shutil.move(jpg,output_dirLR) 
      else: 
        shutil.move(jpg,output_dirHR) 
      '''
      # shutil.move(jpg, output_dir) 
    except IOError as e: 
      print('could not read:',jpg) 
      print('error:',e) 
      print('skip it\n') 
  
  sys.stdout.write('Convert Over!\n') 
  sys.stdout.flush() 
  
  
  
if __name__ == "__main__": 
  current_dir = os.getcwd() 
  print(current_dir) # /Users/gavin/PycharmProjects/pygame
  data_dir = current_dir
  data_dir = '/home/jky/jky/tool/dataprocess_test/TIF2JPG'
  
  image2jpg(data_dir,'jpg')