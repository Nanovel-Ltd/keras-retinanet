
import sys
import os
import argparse

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"
    


def load_model_from_path(model_path):
    model = models.load_model(model_path, backbone_name='resnet50')
    try:
        model = models.convert_model(model)
    except:
        print("Model is likely already an inference model")
    return model


def parse_args(args):
    """ Parse the arguments.
    """
    parser     = argparse.ArgumentParser(description='Simple prediction script for evaluating a RetinaNet network.')
   

    parser.add_argument('--labels-json', help='Path to json file conatining new labels for replacing pascal-voc deafults', type=str, default=None,required=True)
    parser.add_argument('--weights-path', help='path to model weights snapshot', type = str,default= None,required=True)
    parser.add_argument('--min-confidence-cutoff', help="minimum confidence cutoff value, default is 0.5", type=float, default=0.5)
    parser.add_argument("-d", action='store_true')
    parser.add_argument("-o","--output",help="Output dor path for saving result. the result name will be current time",type=str,required=True)
    parser.add_argument("--file-ext",help="imges file type",type=str,default="jpg")
    parser.add_argument("--samples-num",help="number of images to sample from input directory",default=20,type=int,required=False)
    parser.add_argument("input",help="input data path (file/dir)")


    return parser.parse_args(args)

def predict_for_single_image(label_names, model,image_path,confidence_cutoff):
    image = np.asarray(Image.open(image_path).convert('RGB'))
    image = image[:, :, ::-1].copy()

    # copy to draw on
    draw = image.copy()
    draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
    # Image formatting specific to Retinanet
    image = preprocess_image(image)
    image, scale = resize_image(image)


    boxes, scores, labels = model.predict_on_batch(np.expand_dims(image.copy(), axis=0))
    boxes /= scale
    for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
        if score < confidence_cutoff:
            break
    #Add boxes and captions
        color = (255, 255, 255)
        thickness = 2
        b = np.array(box).astype(int)
        cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

        if(label > len(label_names)):
            print("WARNING: Got unknown label, using 'detection' instead")
            caption = "Detection {:.3f}".format(score)
        else:
            caption = "{} {:.3f}".format(label_names[label], score)

        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    return draw

def args_check(args):
    
    if os.path.isdir (args.labels_json) or args.labels_json.split(".")[-1] != "json":
        raise Exception("input labels file must be a json file")
    if args.d:
        if not os.path.isdir(args.output):
            raise Exception("the specified output path is a single file, though the input is a directory")
    else:
        if os.path.isdir(args.output):
            raise Exception("not specified \"-d\" for directory input")

def load_labels_from_json(json_path):
    import json
    with open(json_path,"r+") as json_file:
        labels = json.load(json_file)
    for key in list(labels.keys()):
        labels[int(key)] = labels[key]
        del labels[key]
    return labels
   
def main(args=None):
  
     # parse arguments
    import glob 
    import os
    import random
    import shutil 
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    args_check(args)    
    



    if args.d:
        images_list = glob.glob(os.path.join(args.input,f"*.{args.file_ext}"))
        print("samples num is "+ str(args.samples_num))
        images_list = random.choices(images_list,k=args.samples_num)
        output_path = args.output
        print(f"output path is {output_path}")
        new_output_dir_name = os.path.join(args.output,f"{time.time()}")
        print(f"creating output folder with name: \n{new_output_dir_name}\n _______________________________________")
        os.makedirs(new_output_dir_name)
        print("output folder created successfully!")

        print("loading model....\n\n\n_________________________________")
        model = load_model_from_path(args.weights_path)
        print("**************Model loaded successfully!! ****************\n\n\n")
        print("start batch prediction")
        labels = load_labels_from_json(args.labels_json)
        counter = 0
        for image_path in images_list:
            output_original_name = os.path.join(new_output_dir_name,f"origin_{counter}.{args.file_ext}")
            output_predicted_name = os.path.join(new_output_dir_name,f"predicted_{counter}.{args.file_ext}")
            shutil.copyfile(image_path,output_original_name)
            image_with_boxes = predict_for_single_image(model=model,
                                                        label_names=labels,
                                                        image_path=image_path,
                                                        confidence_cutoff=args.min_confidence_cutoff)

            cv2.imwrite(output_predicted_name,cv2.cvtColor(image_with_boxes,cv2.COLOR_RGB2BGR))
            print(f"predict for image: {image_path}")
            counter+=1
    else:
        print("loading model....\n\n\n_________________________________")
        model = load_model_from_path(args.weights_path)
        print("**************Model loaded successfully!! ****************\n\n\n")
        print("prediction for single file")
        labels = load_labels_from_json(args.labels_json)
        image_with_boxes = predict_for_single_image(model=model,
                                                        label_names=labels,
                                                        image_path=args.input,
                                                        confidence_cutoff=args.min_confidence_cutoff)

        cv2.imwrite(args.output,cv2.cvtColor(image_with_boxes,cv2.COLOR_RGB2BGR))
        print(f"predict for image: {args.input}")
            

if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    args_check(args)
    #imports
    import keras
    import matplotlib.pyplot as plt
    import json
    import time
    from .. import models
    from ..utils.image import preprocess_image, read_image_bgr, resize_image
    import cv2
    import os
    import numpy as np
    import time
    from PIL import Image
    import tensorflow as tf
    main()