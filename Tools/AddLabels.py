# TODO 可以直接把想要检测的类别写入Aclasses列表，然后注释掉下面的while Ture 部分。
#  另外：detect（）只会检测同时包含在Aclasses和classes in model 的类别，所有放心大胆地往Aclasses中塞东西吧
def Addlabels(class_numbers, class_names, Aclasses):
    while True:
        # 想要检测的类别
        while True:
            while True:
                transNum = input(f'\nnumber or name[1/2]：')
                if transNum == '1':
                    Class_name = int(input('input the class you want to detect:'))
                    break
                elif transNum == '2':
                    Class_name = input('input the class you want to detect:')
                    break
            c = Class_name
            print(f'Class_name:{c}')
            bool1 = c in class_numbers
            print(f'bool1:{bool1}')
            bool2 = c in class_names  # 不一定要bool，bl也行（任何字符串转一下就行）
            print(f"bool2:{bool2}")
            # if bool == True:  # Class_name in class_numbers == True一直返回False就很奇怪
            bool = bool1 or bool2
            print(f'输入判别：{bool}')
            if bool1:
                print(f'即将检测含有{c}的图片>_<\n')
                break
            else:
                if bool2:
                    print(f'即将检测含有{class_names[c]}的图片\n')
                    break
                else:
                    print('\n请输入class中有的目标名称或编号')

        if transNum == '1':
            class_input = class_names[int(c)]
            if class_input not in Aclasses:
                Aclasses.append(class_input)
        else:
            if c not in Aclasses:
                Aclasses.append(c)

        enough = input('enough [input y] ?')
        if enough == 'y':
            break
        print(f'{Aclasses},so go on\n')
    return Aclasses


if __name__ == '__main__':
    Aclasses = ['background']  # , 'person', 'car', 'bicycle'
    # TODO 可以直接把想要检测的类别写入Aclasses列表，然后注释掉下面的 Addlabels() 部分。
    #  另外：detect（）只会检测同时包含在Aclasses和classes in model 的类别，所有放心大胆地往Aclasses中塞东西吧
    Class_numbers = {0: 'Red', 1: 'orange-red', 2: 'orange', 3: 'black', 4: 'white', 5: 'orange-yellow',
                     6: 'yellow', 7: 'yellow-green', 8: 'green', 9: 'blue-green', 10: 'blue', 11: 'blue-violet',
                     12: 'violet', 13: 'fuchsia'}
    Class_names = {'Red': 0, 'orange-red': 1, 'orange': 2, 'black': 3, 'white': 4, 'orange-yellow': 5,
                   'yellow': 6, 'yellow-green': 7, 'green': 8, 'blue-green': 9, 'blue': 10, 'blue-violet': 11,
                   'violet': 12, 'fuchsia': 13}
    Addlabels(Class_numbers, Class_names, Aclasses)
