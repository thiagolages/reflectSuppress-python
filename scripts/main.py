import cv2
from reflectSuppress import reflectSuppress
import matplotlib.pyplot as plt
import numpy as np

def show_image(img, title='img', sizex=1280, sizey=720, wait=True, destroy=True):

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    
    if sizex and sizey:
        cv2.resizeWindow(title, sizex, sizey)

    cv2.imshow(title,img)
    
    if wait:
        cv2.waitKey()
    else:
        cv2.waitKey(1)

    if destroy:
        cv2.destroyAllWindows()

if __name__ == '__main__':

    img_path = '../figures/Child_Glass.jpg'

    h = 0.033
    epsilon = 1e-8 
    
    img_extension = img_path.split('/')[-1].split('.')[-1]
    img_name      = img_path.split('/')[-1].split('.')[-2]

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    T = reflectSuppress(img, h, epsilon)

    out_img = cv2.normalize(T, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
    out_img.astype(np.uint8)
    out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
    
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    show_image(img, 'original', wait=False, destroy=False)
    show_image(out_img, 'output')

    cv2.imwrite('../output/'+img_name+'_h'+str(h)+'_eps'+str(epsilon)+'.'+img_extension, out_img)

