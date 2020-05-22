from xml.etree.ElementTree import ElementTree


def read_xml(in_path):
    '''''读取并解析xml文件
    in_path: xml路径
    return: ElementTree'''
    tree = ElementTree()
    tree.parse(in_path)
    return tree


def find_nodes(tree, path):
    '''查找某个路径匹配的所有节点
    tree: xml树
    path: 节点路径'''
    return tree.findall(path)


def change_node_properties(nodelist, kv_map, is_delete=False):
    '''''修改/增加 /删除 节点的属性及属性值
    nodelist: 节点列表
    kv_map:属性及属性值map'''
    for node in nodelist:
        for key in kv_map:
            if is_delete:
                if key in node.attrib:
                    del node.attrib[key]
            else:
                node.set(key, kv_map.get(key))


def get_node_by_keyvalue(nodelist, kv_map):
    '''''根据属性及属性值定位符合的节点，返回节点
    nodelist: 节点列表
    kv_map: 匹配属性及属性值map'''
    result_nodes = []
    for node in nodelist:
        if if_match(node, kv_map):
            result_nodes.append(node)
    return result_nodes


def if_match(node, kv_map):
    '''''判断某个节点是否包含所有传入参数属性
    node: 节点
    kv_map: 属性及属性值组成的map'''
    for key in kv_map:
        if node.get(key) != kv_map.get(key):
            return False
    return True


def write_xml(tree, out_path):
    '''
        将xml文件写出
        tree: xml树
        out_path: 写出路径
    '''
    tree.write(out_path, encoding="utf-8", xml_declaration=True)


if __name__ == '__main__':
    tree = read_xml(r'F:\upload\1.xml')
    root = tree.getroot()
    root.set("abc", '1')
    num = 100
    for i in range(num):

        ip = '100.0.{0}.{1}'.format(int(i / 250), i % 250)
        root.set("abc", str(i))
        root.set('ip',ip)
        file = r'F:\upload\1\{0}.xml'.format(i)
        print(file)
        write_xml(tree, file)

        # change_node_properties(tree, {"abc": "1", 'ip':'1.2.3.4'})
