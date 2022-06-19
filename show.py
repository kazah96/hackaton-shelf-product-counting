from PIL import Image
from matplotlib import pyplot


shelf_file1 = 'db1441.jpg' 
query_file2 = 'qr66.jpg'


fig, m_axs = pyplot.subplots(2, 2, figsize=(16, 16))

img1 = Image.open('datasets/PrivateTestSet/shelves/' + shelf_file1)
img2 = Image.open('datasets/PrivateTestSet/queries/' + query_file2)
 

m_axs[0,0].imshow(img1)
m_axs[0,1].imshow(img2)

# pyplot.imshow(img1)
# pyplot.imshow(img2)

pyplot.waitforbuttonpress()