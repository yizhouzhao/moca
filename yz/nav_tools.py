import numpy as np
import copy

from ai2thor.server import Event
from ai2thor.controller import Controller

# class SegInfo:
#     def __init__(self):
#         self.seg_frame = None
#         self.

class FrameInfo():
    '''
    record segmentation frame info
    '''
    candidates = None
    def __init__(self, event:Event):
        self.agent_info = event.metadata["agent"].copy()
        self.instance_detections2D = event.instance_detections2D
        self.object_info = [{"objectId": item["objectId"],
                             "objectType": item["objectType"],
                             "visible": item["visible"],
                             "distance": item["distance"]
        } for item in event.metadata["objects"]]

        # get objtype 2 index
        self.objType2Index = {}
        for i in range(len(self.object_info)):
            if self.object_info[i]["objectType"] not in self.objType2Index:
                # only get the first item
                self.objType2Index[self.object_info[i]["objectType"]] = i

        # get scene obj types
        self.all_object_types = []
        #self.all_objectType2Index = {}
        for i, item in enumerate(self.instance_detections2D):
            obj_t = item.split("|")[0]
            if not obj_t.startswith("FP"):
                self.all_object_types.append(obj_t)
                # if obj_t not in self.all_objectType2Index:
                #     self.all_objectType2Index[obj_t] = i
    
    @staticmethod
    def build_candidates(my_candidates:list):
        FrameInfo.candidates = my_candidates

    def get_answer_array_for_one_candidate(self, cand):
        if cand not in self.all_object_types:
            cand_array = [-1,0,0]
        else:
            # if candidates in the array
            if cand not in self.objType2Index:
                # if it is structure object
                cand_array = [1, 0, 0]
            else:
                info = self.object_info[self.objType2Index[cand]]
                # if it a normal object
                dist = info["distance"]
                visible = 1 if info["visible"] else 0
                cand_array = [1, dist, visible]
        
        return cand_array
    
    def get_answer_array_for_all_candidates(self, indicates:list):
        '''
        Get the answer array the query of all object types indicating by a bool array
        input:
            indicates: a bool array of the same length as candidates
        output:
            array: len(indicates) x question_num
        '''
        answer_array = []
        assert len(indicates) == len(FrameInfo.candidates)
        for i, query_or_not in enumerate(indicates):
            if query_or_not:
                cand_array = self.get_answer_array_for_one_candidate(FrameInfo.candidates[i])
            else:
                cand_array = [0, 0, 0]
            
            answer_array.extend(cand_array)

        return answer_array




# def get_event_info(event:Event, only_obj=False):
#     # agent
#     agent_info = event.metadata["agent"].copy()

#     # frame 
#     seg_frame = event.instance_segmentation_frame
#     seg_frame_reshaped = seg_frame.reshape((-1,3))
#     image_colors = np.unique(seg_frame_reshaped, axis=0)
    
#     all_objects = event.metadata['objects']

#     seg_objects_info = {}
#     seg_object_types = []
#     for color_key in event.color_to_object_id:
#         for i in range(len(image_colors)):
#             if color_key[0] == image_colors[i][0] and color_key[1] == image_colors[i][1] and color_key[2] == image_colors[i][2]:
#                 object_id = event.color_to_object_id[color_key]
                
#                 # get object type
#                 #  if only consider objects without structure(floor wall)
#                 if only_obj:
#                     for obj in all_objects:
#                         if object_id == obj["objectId"]:
#                             seg_object_types.append(obj["objectType"])
#                             break
#                 else:
#                     object_type = object_id.split("|")[0]
#                     if not object_type.startswith("FP"):
#                         seg_object_types.append(object_type)

#                 seg_objects_info[object_id] = {"distance": 10, "box": [-1, -1, -1, -1]}

#                 # get object distance from player
#                 for obj in all_objects:
#                     if object_id == obj["objectId"]:
#                         dist = distance_between_position(agent_info["position"], obj["position"])
#                         seg_objects_info[object_id]["distance"] = dist
#                         break
                
       
#                 break

#     return seg_frame, seg_objects_info, seg_object_types
    

def answer_object_info_from_segframe(objectType, seginfo):
    obj_exist = objectType in seginfo[2]
    return obj_exist
    

def get_all_objtype_in_event(event: Event):
    all_objtype = []
    for obj in event.metadata['objects']:
        if obj["objectType"] not in all_objtype:
            all_objtype.append(obj["objectType"])

    return all_objtype

def get_all_objtype_in_room(room_type:str):
    '''
    Get all Object Types in a certern room type: kitchen, living_room, bedroom, bathroom
    '''
    if room_type == "kitchen":
        room_start_index = 0
    elif room_type == "living_room":
        room_start_index = 200
    elif room_type == "bedroom":
        room_start_index = 300
    else:
        room_start_index = 400

    all_obj_type = []

    controller = Controller(scene="FloorPlan1", 
                        renderInstanceSegmentation=True,
                        width=1080,
                        height=1080)
    
    for i in range(1, 31):
        controller.reset(scene="FloorPlan" + str(room_start_index + i))
        event = controller.step("Done")
        all_obj = get_all_objtype_in_event(event)
        for objtype in all_obj:
            if objtype not in all_obj_type:
                all_obj_type.append(objtype)

    controller.stop()
    return all_obj_type

    