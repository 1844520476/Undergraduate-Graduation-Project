def TextAdd():
    """
    使用add可以方便的对文本信息进行统一修改，
    使格式与训练数据集中的文本图像对更匹配
    """
    text_add = input('\n需要对【文本前后增补】吗？[y/n]:')
    if text_add == 'y':
        print(f'示例：add1 + Label_list[i] + add2（记得打前后空格）')
        add1 = input('add1:')
        add2 = input('add2:')
        return add1, add2
    else:
        add1 = ''
        add2 = ''
        return add1, add2