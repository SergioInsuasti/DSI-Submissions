# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 02:37:25 2019
Find the Area of the Lot Area and The Built up Area
@author: Rajas
"""

import numpy as np
import pandas as pd
import requests
from time import sleep
import cv2
from skimage.measure import regionprops
import math as Math


# Get the data
df =pd.read_csv('../Datasets/INPUT/NSW Data Complete Balcktown and surrounding suburbs.csv')
df.dropna(subset = ['latitude'], inplace = True)
# Only keep Properties labeled house
df2 = df[df.type == 'house']
# Seperate the data having data on land sqm. This is main dataset. Those having no landsqm is test dataset.
data = df2[df2.land_sqm.notnull()]
test = df2[df2.land_sqm.isnull()]

''' Important : - Change the api_key for your account. Then call this function
to grab the google image data and save the image files'''

def getimagegoogle (data,zoom = '19'):
    for i in range(data.shape[0]):
        baseurl ='https://maps.googleapis.com/maps/api/staticmap?'
        location = str(data.latitude.iloc[i])+','+str(data.longitude.iloc[i])
        size = '400x400'
        api_key = 'AIzaSyB-XFzW6yp3jOskOQb5RIWk-nWQISpZ3NA'
        maptype = 'roadmap'
        final_url =baseurl+'center='+location+'&size='+size+'&zoom='+zoom+'&maptype='+maptype+'&key='+api_key
               
        sleep(0.1)
        api_response = requests.get(final_url)
        

        fname = 'sell_id_'+str(data.id.iloc[i])+'.png'
        with open(fname,'wb') as f:
            f.write(api_response.content)
            
# Uncomment the next line only when grabbing the images. Then comment it back.            
getimagegoogle(data)
            
''' 
Do the Image Processing on the files to get the lot and built up area. 
'''

# Load the File.
no_mark_image_path = 'sell_id_' + str(data.id.iloc[0])+'.png' 
nmi = cv2.imread(no_mark_image_path)
gray = cv2.cvtColor(nmi, cv2.COLOR_BGR2GRAY)

'''
 Display the grayscale image - For testing / Debugging purpose uncomment 
 following learning. The program won't run further till some key is pressed 
 and the image is closed.
'''
cv2.imshow('Gray image', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

''' 
Modeling - Create the Mask based on the grayscale color of the built up area.
'''
centroid = [200,200]           # The centroid is 200,200 for a 400 X 400 Image.
lower_grey = np.array([241])
higher_grey = np.array([248])
mask = cv2.inRange(gray,lower_grey,higher_grey)
mask = 255-mask # inversion
zoom = 19 # Make sure it is the correct value as defined in image grab func
meters_ppx=156543.03392*Math.cos(data.latitude[0]*Math.pi/180)/Math.pow(2,zoom)
actual_lot_area = data.land_sqm[0]
'''
 Display the mask image - For testing / Debugging purpose uncomment 
 following lines. The program won't run further till some key is pressed 
 and the image is closed.
'''
cv2.imshow('Mask image',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()
''' 
Label and show the connected components in the  mask
'''
ret, labels = cv2.connectedComponents(mask)

'''
 Display the connected image - For testing / Debugging purpose uncomment 
 following lines. The program won't run further till some key is pressed 
 and the image is closed. Should be avoided in normal run.
'''

# Map component labels to hue val
#label_hue = np.uint8(179*labels/np.max(labels))
#blank_ch = 255*np.ones_like(label_hue)
#labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
#
## cvt to BGR for display
#labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
#
## set bg label to black
#labeled_img[label_hue==0] = 0
#
#cv2.imshow('labeled.png', labeled_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


# Function to find the label containing centroid (200,200)
def label_centroid (lab_img,centroid):
    
    for i in range(len(lab_img)):
        cord = lab_img[i].coords
        if [centroid[0]] in cord[:,0]:
            idx =np.where(cord[:,0]==centroid[0])
            if [centroid[1]] in cord [idx,1]:
                return i


'''
Label the regions and find which label contains the centroid.
'''
labeled_image = regionprops(labels)

build_label = label_centroid(labeled_image,centroid)
#label_cord = labeled_image[build_label].coords # Breeks the code dont use.
predicted_builtup_area = (labeled_image[build_label].area)*meters_ppx**2

''' According to the google employee following formula may give meters/px 
in a map 

metersPerPx=156543.03392*Math.cos(latLng.lat()*Math.PI/180)/Math.pow(2,zoom) 

https://groups.google.com/forum/#!topic/google-maps-js-api-v3/hDRO4oHVSeM
'''


''' Alternate Way - Get the number of pixels for lot area excluding the inner 
built up area. Then invert the image and get the number of pixels in the built
up area. Add the two Areas to get the total number of pixels in the lot area. 
Finally, get the actual lot area and divide it by total number of pixels to 
get the multiplying factor. Then Calculate the built up area my taking product
of the multiplying factor and number of pixels in built up area.'''


'''
Calculate the Lot Size 
'''

# Convert the labeled built up part to 0 to merge it into lot size

coords = [tuple(item) for item in label_cord]
for loc in range(len(coords)):
    labels[coords[loc]] = 0
'''
The home should be vanished in next visualization
'''    
## Map component labels to hue val
#label_hue = np.uint8(179*labels/np.max(labels))
#blank_ch = 255*np.ones_like(label_hue)
#labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
#
## cvt to BGR for display
#labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
#
## set bg label to black
#labeled_img[label_hue==0] = 0
#
#cv2.imshow('labeled.png', labeled_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Invert the Image and relabel to get the Lot size area
for loc in range(len(coords)):
    mask[coords[loc]] = 0

mask = 255 - mask   
cv2.imshow('Home Removed image',mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Label and calculate the area
ret, labels_lot = cv2.connectedComponents(mask)
labeled_image_lot = regionprops(labels_lot)
build_label_lot = label_centroid(labeled_image_lot,centroid)
label_cord_lot = labeled_image[build_label_lot].coords
predicted_lot_area = (labeled_image_lot[build_label_lot].area)*meters_ppx**2