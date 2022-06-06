# TODO 这是用于目标检测（去冗余/定位模块）的标签：CustomLabel if you want to detect custom label classes,
#  you need 1.class_names_custom & 2.class_numbers_custom (of 3.course:custom.pt)

classes_custom = [
    'happy', 'angry', 'fear', 'surprise', 'sad',
    'anxiety', 'awe', 'embarrassment', 'boredom',
    'calm', 'confusion', 'contempt', 'disappointment',
    'disgust', 'pride', 'jealousy'
] # 辅助用

# TODO 在本模块运行后，将终端输出替换下面
class_numbers_custom = {'face': 0}
class_names_custom = {0: 'face'}

face_numbers = {'face': 0}
face_names = {0: 'face'}

if __name__ == '__main__':
    # class_numbers_custom = {}
    # class_names_custom = {}
    for i in range(len(classes_custom)):
        CLASS = classes_custom[i]
        class_names_custom[i] = CLASS
        class_numbers_custom[CLASS] = i
    print(f'class_numbers_custom:{class_numbers_custom}\nclass_names_custom:{class_names_custom}')

"""
for example:

classes_custom = ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale','scratches']

class_names2 = {'crazing': 0, 
                        'inclusion': 1, 
                        'patches': 2, 
                        'pitted_surface': 3, 
                        'rolled-in_scale': 4,
                        'scratches': 5
                          }
                
class_numbers2 = {0: 'crazing', 
                        1: 'inclusion', 
                        2: 'patches', 
                        3: 'pitted_surface', 
                        4: 'rolled-in_scale',
                        5: 'scratches'
                            }
"""
