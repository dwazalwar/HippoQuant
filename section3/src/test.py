
import os
import sys
import datetime
import time
import shutil
import subprocess

import numpy as np
import pydicom

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def create_report(inference, header, orig_vol, pred_vol):
    """Generates an image with inference report

    Arguments:
        inference {Dictionary} -- dict containing anterior, posterior and full volume values
        header {PyDicom Dataset} -- DICOM header
        orig_vol {Numpy array} -- original volume
        pred_vol {Numpy array} -- predicted label

    Returns:
        PIL image
    """

    # The code below uses PIL image library to compose an RGB image that will go into the report
    # A standard way of storing measurement data in DICOM archives is creating such report and
    # sending them on as Secondary Capture IODs (http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_A.8.html)
    # Essentially, our report is just a standard RGB image, with some metadata, packed into 
    # DICOM format. 

    pimg = Image.new("RGB", (1000, 1000))
    draw = ImageDraw.Draw(pimg)

    header_font = ImageFont.load_default()
    main_font = ImageFont.load_default()

    slice_nums = [orig_vol.shape[2]//3, orig_vol.shape[2]//2, orig_vol.shape[2]*3//4] # is there a better choice?

    # TASK: Create the report here and show information that you think would be relevant to
    # clinicians. A sample code is provided below, but feel free to use your creative 
    # genius to make if shine. After all, the is the only part of all our machine learning 
    # efforts that will be visible to the world. The usefulness of your computations will largely
    # depend on how you present them.

    # SAMPLE CODE BELOW: UNCOMMENT AND CUSTOMIZE
    draw.text((10, 0), "HippoVolume.AI", (255, 255, 255), font=header_font)
    draw.multiline_text((10, 90),
                        f"Patient ID: {header['PatientID']}\n"
                        f"Posterior Volume: {inference['posterior']}\n"
                        f"Anterior Volume: {inference['anterior']}\n"
                        f"Total Hippocampal Volume: {inference['total']}\n",
                        fill=(255, 255, 255), font=main_font)

    # STAND-OUT SUGGESTION:
    # In addition to text data in the snippet above, can you show some images?
    # Think, what would be relevant to show? Can you show an overlay of mask on top of original data?
    # Hint: here's one way to convert a numpy array into a PIL image and draw it inside our pimg object:
    #
    # Create a PIL image from array:
    # Numpy array needs to flipped, transposed and normalized to a matrix of values in the range of [0..255]
    # nd_img = np.flip((slice/np.max(slice))*0xff).T.astype(np.uint8)
    # This is how you create a PIL image from numpy array
    # pil_i = Image.fromarray(nd_img, mode="L").convert("RGBA").resize(<dimensions>)
    # Paste the PIL image into our main report image object (pimg)
    # pimg.paste(pil_i, box=(10, 280))

    # Normalize original slice to [0, 255] and convert to uint8
    draw.text((10, 280), f"Showing slices at {slice_nums[0]}, {slice_nums[1]}, {slice_nums[2]}", (255, 255, 255), font=header_font)
    
    spacing = 20
    num_panels = 3
    total_spacing = spacing * (num_panels - 1)
    panel_width = (1000 - total_spacing) // num_panels
    panel_height = 750 - 300  # top space for text

    for idx, slice_ in enumerate(slice_nums):
        orig_slice = orig_vol[:, :, slice_]
        pred_mask = pred_vol[:, :, slice_]

        # Normalize and prepare original slice
        orig_img = np.flip((orig_slice / np.max(orig_slice)) * 255).T.astype(np.uint8)
        pil_orig = Image.fromarray(orig_img, mode="L").convert("RGBA").resize((panel_width, panel_height))

        # Prepare overlay mask
        mask_overlay = np.zeros((*pred_mask.shape, 4), dtype=np.uint8)
        mask_overlay[pred_mask == 1] = [255, 0, 0, 120]   # Anterior - red
        mask_overlay[pred_mask == 2] = [0, 255, 0, 120]   # Posterior - green
        pil_mask = Image.fromarray(mask_overlay, mode="RGBA").resize((panel_height, panel_height))

        # Overlay mask onto slice
        composite = pil_orig.copy()
        composite.paste(pil_mask, (0, 0), mask=pil_mask)

        # Paste into the report image at the appropriate location
        x_offset = idx * (panel_width + spacing)
        pimg.paste(composite, (x_offset, 300))

    return pimg

header = {"PatientID": "12345"}
inference = {"posterior": 1250.3, "anterior": 987.6, "total": 2237.9}

orig_slice = np.random.rand(128, 128, 60)  # Simulated MRI slice
pred_mask = np.random.randint(0, 3, (128, 128, 60))  # Example segmentation mask

pimg = create_report(inference, header, orig_slice, pred_mask)
pimg.save("inference_report.png")
pimg.show()
