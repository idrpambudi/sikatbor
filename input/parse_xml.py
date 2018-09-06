import os
import xml.etree.ElementTree as ET

def get_danger_frame(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    result = set()

    for track in root.iter('track'):
        for box in track.iter('box'):
            frame_no = int(box.attrib['frame'])
            frame_sample_no = int(frame_no/10) + 1
            result.add(frame_sample_no)

    return result

current_dir = os.path.dirname(os.path.realpath(__file__))
xml_dir = current_dir + "/xml-labels/"
frame_samples_dir = current_dir + "/train/"
labels_csv = open(current_dir + '/labels.csv', 'w')
labels_csv.write('filename,frame_number,class\n')
frames_dict = {}

for file in sorted(os.listdir(xml_dir)):
    filename = os.fsdecode(file)
    vid_name = filename[0:-4]

    if filename.endswith(".xml"):
       danger_frames = get_danger_frame(xml_dir + filename)
       frames_dict[vid_name] = danger_frames

for file in sorted(os.listdir(frame_samples_dir)):

    filename = os.fsdecode(file)
    if not filename.endswith(".jpg"):
        continue

    filename_no_ext = filename[0:-4]
    split_filename = filename_no_ext.split('_')
    vid_name = split_filename[0]
    frame_sample_no = int(split_filename[1])
    real_frame_no = (frame_sample_no * 10) - 9
    label = 0

    try:
        danger_frames_set = frames_dict[vid_name]
    except:
        print('XML file does not exists.')
        labels_csv.write('{},{},{}\n'.format(filename, real_frame_no, label))
        continue
    
    if frame_sample_no in danger_frames_set:
        label = 1

    labels_csv.write('{},{},{}\n'.format(filename, real_frame_no, label))
