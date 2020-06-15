ORIGINAL_NAME = 'original' # Original Image without processing
JPEG_NAME = 'jpeg' # JPEG compressed only
DEMOSAIC_NAME = 'demosaic' # Bayer-Demosaicing only
BOTH_NAME = 'both' # Demosaicing + JPEG compressed
IMAGING_OPERATIONS = [ORIGINAL_NAME, JPEG_NAME, DEMOSAIC_NAME, BOTH_NAME]

from colour_demosaicing import (
    demosaicing_CFA_Bayer_Malvar2004,
    demosaicing_CFA_Bayer_bilinear)
demosaic_func_dict = {
    'Malvar2004' : demosaicing_CFA_Bayer_Malvar2004,
    'bilinear' : demosaicing_CFA_Bayer_bilinear,
}

RESNET_MODELS = ["resnet50","resnet101",]
