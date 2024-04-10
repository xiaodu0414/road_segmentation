import cv2
import numpy as np
from mmcv import Config
import multiprocessing
import time
from mmseg.apis import inference_segmentor, init_segmentor

def update_info(mask,frame,class_names,preds,scale):

    contours, hierarchy = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)     #提取轮廓
    ori_h, ori_w, _ = frame.shape
    for i, contour in enumerate(contours): # 根据轮廓获取坐标和类别
        x, y, w, h = cv2.boundingRect(contour)
        category = class_names[preds[y:y+h,x:x+w][0][0]]
        if category == '0':
            # 将坐标从调整后的图像大小还原到原始图像大小
            x_ori, y_ori, w_ori, h_ori = int(x * ori_w / mask.shape[1]), int(y * ori_h / mask.shape[0]), \
                                        int(w * ori_w / mask.shape[1]), int(h * ori_h / mask.shape[0])
            # print(category)
            for point in contour:
                    x_ori, y_ori = int(point[0][0] * ori_w / mask.shape[1]*scale), int(point[0][1] * ori_h / mask.shape[0]*scale)
                    # print(x_ori,y_ori)

def maskVisualization(frame,preds):
     # 将分割结果可视化，并使用 OpenCV 显示推理结果
    palette = [[0, 0, 0], [150,150, 0]]
    palette = np.array(palette, dtype=np.uint8)
    masked_image = palette[preds]
    mask = cv2.threshold(masked_image[:, :, 0], 1, 255, cv2.THRESH_BINARY)[1]
    mask_inv = cv2.bitwise_not(mask)
    # 调整掩膜 mask_inv 的大小，使之与原图 image 的大小一致
    mask_inv = cv2.resize(mask_inv, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    bg = cv2.bitwise_and(frame, frame, mask=mask_inv)
    fg = cv2.bitwise_and(masked_image, masked_image, mask=mask)
    fg = cv2.resize(fg, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)

    alpha = 0.6
    result_image = cv2.addWeighted(fg, alpha, bg,1-alpha, 1)#叠加
    return result_image,mask


def segformer(Config_file,Checkpoint_file,Resized_w, Resized_h,url,index):

    config_file = Config_file
    checkpoint_file = Checkpoint_file
    cfg = Config.fromfile(config_file)
    resized_w,resized_h = Resized_w,Resized_h           # 缩放尺寸 640，480
    image_path = url
    num_classes = cfg.model.decode_head.num_classes
    class_names = [str(i) for i in range(num_classes)]
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    video = cv2.VideoCapture(image_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    if fps != fps:  # 检测到实际帧速率为 NaN
        print("无法获取帧速率信息")

    else:
        print("id为: {} 的帧速率为: {} ".format(index,fps))
    count = 0  #抽帧
    while True:
        ret,image = video.read()
        if not ret:
            break
        count += 1 #每读一帧+1
        if count %5 == 0:
            Height,Width = image.shape[0],image.shape[1]                                        #原图宽高
            scale = min(Width / resized_w, Height / resized_h)                                  #计算缩放比
            image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR) ## 调整图像大小
            start = time.time()
            result = inference_segmentor(model, image)                                 # 将 SegFormer 模型应用于测试图像进行前向推理，得到分割结果
            end = time.time()
            print("infer use time is :{}ms".format(end-start))
            preds = result[0]                                                                                                   # print("预测结果\n",preds)

            # for i in range(len(class_names)):
            #     if i in preds:
            #         print('{}: {}'.format(i, class_names[i]))

            result_image,mask = maskVisualization(image,preds)
            update_info(mask,image,class_names,preds,scale)

            cv2.imshow('image', result_image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    video.release()
    cv2.destroyAllWindows()

def main():

    Resized_w, Resized_h = 640, 480
    config_file = 'local_configs/segformer/B0/segformer.b0.1024x1024.city.160k.py'
    checkpoint_file = 'segformer/latest.pth'
    # image_path = 'demo/vlcsnap-2023-05-12-15h44m24s278.jpg' 
    # cap = 'rtsp://admin:123qweasd@192.168.1.239:554/h264/ch1/main/av_stream'
    urls = [
         'rtsp://admin:123qweasd@192.168.1.246:554/h264/ch1/main/av_stream',
         'rtsp://admin:123qweasd@192.168.1.238:554/h264/ch1/main/av_stream'
    ]
    processes = []
    # for i, url in enumerate(urls, start=1):

    for index,url in enumerate(urls,start=1):
        p = multiprocessing.Process(target=segformer, args=(config_file,checkpoint_file,Resized_w,Resized_h,url,index))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()



if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    main()
