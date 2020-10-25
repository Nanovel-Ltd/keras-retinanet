
import sys
import os
import argparse

# Allow relative imports when being executed as script.
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
    import keras_retinanet.bin  # noqa: F401
    __package__ = "keras_retinanet.bin"
    

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

def load_model_from_path(model_path):
    model = models.load_model(model_path, backbone_name='resnet50')
    try:
        model = models.convert_model(model)
    except:
        print("Model is likely already an inference model")
    return model




def old_func():

    confidence_cutoff = 0.3
    model1_path = "D:/Repositories/valencia-training/keras-retinanet/snapshots/resnet50_coco_best_v2.1.0.h5"
    model1_classes = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

    model2_path = "D:/Repositories/valencia-training/models/valencia_iference.h5"
    model2_classes = {0: 'hidden orange'}


    #load models
    model1 = load_model_from_path(model1_path)
    model2 = load_model_from_path(model2_path)

    #load images list
    import glob
    import os
    import random
    images_dir = "D:\ImagesFromOrchardSorted\lastSet"
    images_list = glob.glob(os.path.join(images_dir,"*.png"))
    images_list = random.choices(images_list,k=30)
    print ("images:/n___________________________________________ ")
    print (images_list)
    output_dir_name =f"D:/Repositories/valencia-training/keras-retinanet/examples/{ time.time()}"
    os.mkdir(output_dir_name)
    results_from_model1 = []
    results_from_model2 = []
    # images_list = ["D:/Repositories/valencia-training/valenciaVOC/JPEGImages/Batch1_(14).jpg"]
    result_counter = 0
    for image_path in images_list:
        image = np.asarray(Image.open(image_path).convert('RGB'))
        image = image[:, :, ::-1].copy()

    # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        draw1 = draw.copy()
        draw2 = draw.copy()
        # Image formatting specific to Retinanet
        image = preprocess_image(image)
        image, scale = resize_image(image)


        
        # image_rgb = cv2.cvtColor(image_data,cv2.COLOR_RGB2BGR)
        counter = 0
        
        boxes1, scores1, labels1 = model1.predict_on_batch(np.expand_dims(image.copy(), axis=0))
        boxes1 /= scale
        for box, score, label in zip(boxes1[0], scores1[0], labels1[0]):
        # scores are sorted so we can break
            if score < confidence_cutoff:
                break
            if label == 49:
                counter+=1
        #Add boxes and captions
            color = (255, 255, 255)
            thickness = 2
            b = np.array(box).astype(int)
            cv2.rectangle(draw1, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

            if(label > len(model1_classes)):
                print("WARNING: Got unknown label, using 'detection' instead")
                caption = "Detection {:.3f}".format(score)
            else:
                caption = "{} {:.3f}".format(model1_classes[label], score)

            cv2.putText(draw1, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
            cv2.putText(draw1, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


        results_from_model1+=[counter]

        boxes2, scores2, labels2 = model2.predict_on_batch(np.expand_dims(image.copy(), axis=0))
        boxes2 /= scale

        counter=0

        for box, score, label in zip(boxes2[0], scores2[0], labels2[0]):
        # scores are sorted so we can break
            if score < confidence_cutoff:
                break
            #Add boxes and captions
            color = (255, 255, 255)
            thickness = 2
            b = np.array(box).astype(int)
            cv2.rectangle(draw2, (b[0], b[1]), (b[2], b[3]), color, thickness, cv2.LINE_AA)

            if(label > len(model2_classes)):
                print("WARNING: Got unknown label, using 'detection' instead")
                caption = "Detection {:.3f}".format(score)
            else:
                caption = "{} {:.3f}".format(model2_classes[label], score)

            cv2.putText(draw2, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
            cv2.putText(draw2, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

            if label == 0:
                counter+=1
        cv2.imwrite(os.path.join(output_dir_name,str(result_counter)+"_coco.jpg"),cv2.cvtColor(draw1,cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(output_dir_name,str(result_counter)+"_trained.jpg"),cv2.cvtColor(draw2,cv2.COLOR_RGB2BGR))
        result_counter+=1
        results_from_model2+=[counter]
        # f, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(draw1)
        # ax1.set_title('coco model')
        # ax2.imshow(draw2)
        # ax2.set_title("our model")
        # plt.show()
        



    results = {"model1":results_from_model1,
                "model2":results_from_model2}
    file_name = os.path.join(output_dir_name,"data.json")
    with open(file_name,"w+") as json_file:
        json.dump(results,json_file)

    f = plt.figure()
    plt.plot(results_from_model1)
    plt.plot(results_from_model2)
    plt.show()
    f.savefig(os.path.join(output_dir_name,"results.pdf"), bbox_inches='tight')

    plt.waitforbuttonpress()


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
    import sys
    if args is None:
        args = sys.argv[1:]
    args = parse_args(args)
    
    args_check(args)
    
    
    



    if args.d:
        import glob
        import os
        import random
        import shutil
        images_list = glob.glob(os.path.join(args.input,f"*.{args.file_ext}"))
        print("samples num is "+ str(args.samples_num))
        images_list = random.choices(images_list,k=args.samples_num)
        output_path = args.output
        print(f"output path is {output_path}")
        new_output_dir_name = os.path.join(args.output,f"{time.time()}")
        print(f"creating output folder with name: \n{new_output_dir_name}\n _______________________________________")
        os.makedirs(new_output_dir_name)
        print("output older created successfully!")

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
            

if __name__ == "__main__":
    main()